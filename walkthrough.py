"""
逐步演示 Atlas-mini 内部在做什么（本机窗口，按键翻页，像讲课一样一步步展开）。

    python walkthrough.py              # 打开窗口
    python walkthrough.py --scene 2    # 换一个测试场景
    python walkthrough.py --export     # 不开窗口，把每一步的关键帧存到 out/walkthrough/
    python walkthrough.py --gif        # 不开窗口，把每一步录成 GIF，存到 out/gifs/
    python walkthrough.py --lang en    # 画面文字改成英文（GIF / 导出存到 out/gifs_en/、out/walkthrough_en/）

按键：→ 或 空格 下一步 · ← 上一步 · R 重播当前步骤的动画 · Q 退出

    第 1 步  世界和相机：一个训练时从没见过的场景，两台相机拍了照片
    第 2 步  相机 → Plücker 射线：每个像素变成一条 3D 光线
    第 3 步  矩 m = o × d：光线"在哪里"（离原点多远、在哪个平面里）
    第 4 步  切成 token：3 张图 → 192 个 token 的序列
    第 5 步  注意力：目标视角的一个位置，在上下文照片里"看"哪里
    第 6 步  去噪：从纯噪声到新视角
    第 7 步  自回归：只给 1 张照片，绕场景一圈，每张生成的图都接回上下文
    第 8 步  长出 3D：生成的深度 → 点云
"""

import argparse
import math
import os
import sys
import threading
import time

import matplotlib
import numpy as np
import torch

import scenes as S
from demo import OUT, Scene, cameras, load_model, psnr, rgb  # 注意：demo.py 会把后端设成 Agg

# 必须在导入 demo 之后再设后端，否则会被 demo.py 的 Agg 覆盖，窗口打不开
if "--export" in sys.argv or "--gif" in sys.argv:
    matplotlib.use("Agg")
elif sys.platform == "darwin":
    matplotlib.use("macosx")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang HK", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
for k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.quit"):
    plt.rcParams[k] = []  # 这些键留给翻页

# 画面语言：--lang en 为英文，默认中文。步骤标题在导入时就要用到，所以直接从命令行读取
LANG = "en" if any(a in ("--lang=en",) for a in sys.argv) or \
    ("--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1:][:1] == ["en"]) else "zh"


def L(zh: str, en: str) -> str:
    return en if LANG == "en" else zh


CTX_COLORS = ["#e53e3e", "#3182ce"]
TGT_COLOR = "#38a169"
N_STEPS = 40  # 去噪步数
CFG = 2.0


# ═══════════════════════════════ 预计算（启动时一次性算完） ═══════════════════════════════


def world_cloud(scene, dev, n_max=5000):
    """从 12 个视角的真实 RGB-D 反投影出整个场景的点云，只用于画"世界长什么样"。"""
    R, eye = cameras(list(range(0, 360, 30)), elev_deg=35, device=dev)
    views = scene.shoot(R, eye)
    pts, cols = zip(*[S.backproject(views[k].cpu(), R[k].cpu(), eye[k].cpu()) for k in range(12)])
    pts, cols = torch.cat(pts), torch.cat(cols)
    keep = (pts[:, :2].abs() < 2.3).all(1) & (pts[:, 2] < 2.5)
    pts, cols = pts[keep], cols[keep]
    idx = torch.randperm(len(pts), generator=torch.Generator().manual_seed(0))[:n_max]
    return pts[idx].numpy(), cols[idx].numpy()


def pick_query_patches(gt, n=6):
    """挑目标视角里落在物体上的 patch（颜色饱和、有深度），彼此隔开。"""
    col = gt[:3].cpu()
    sat = (col.max(0).values - col.min(0).values).unfold(0, 4, 4).unfold(1, 4, 4).mean((-1, -2))
    finite = torch.isfinite(S.decode_depth(gt[3].cpu())).float().unfold(0, 4, 4).unfold(1, 4, 4).mean((-1, -2))
    cand = [(sat[y, x].item(), y, x) for y in range(8) for x in range(8) if sat[y, x] > 0.35 and finite[y, x] == 1]
    picked = []
    for _, y, x in sorted(cand, reverse=True):
        if all(abs(y - py) + abs(x - px) >= 3 for py, px in picked):
            picked.append((y, x))
        if len(picked) == n:
            break
    return picked


def correspondence(gt, tgt_R, tgt_eye, ctx_R, ctx_eye, py, px):
    """目标 patch 中心那个像素的真实 3D 点，投到每个上下文相机里的像素坐标（看不到则为 None）。"""
    v, u = py * 4 + 2, px * 4 + 2
    z = S.decode_depth(gt[3].cpu())[v, u]
    o, d = S.pixel_rays(tgt_R.cpu(), tgt_eye.cpu())
    X = o[0] + z * d[v * S.IMG + u]
    out = []
    for c in range(len(ctx_R)):
        R = ctx_R[c].cpu()
        pc = R @ X - R @ ctx_eye[c].cpu()
        uu, vv = (pc[:2] / pc[2] * S.FOCAL + S.IMG / 2).tolist()
        out.append((uu, vv) if pc[2] > 0 and 0 <= uu < S.IMG and 0 <= vv < S.IMG else None)
    return out


@torch.no_grad()
def denoise(model, ctx, ctx_R, ctx_eye, tgt_R, tgt_eye, dev, seed=0, attn_at=None):
    """带记录的去噪：每一步的 x_t、模型对 x₀ 的估计；attn_at 指定在哪一步记录最后一层的注意力。"""
    n = ctx.shape[0]
    ctx4 = torch.cat([ctx, torch.zeros(4 - n, 4, S.IMG, S.IMG, device=dev)])[None]
    rays4 = torch.cat([S.plucker(ctx_R, ctx_eye), torch.zeros(4 - n, 6, S.IMG, S.IMG, device=dev)])[None]
    valid = (torch.arange(4, device=dev) < n)[None]
    tgt_rays = S.plucker(tgt_R, tgt_eye)
    torch.manual_seed(seed)
    x = torch.randn(1, 4, S.IMG, S.IMG, device=dev)
    frames, attn = [], None
    for i in range(N_STEPS):
        t = torch.full((1,), 1.0 - i / N_STEPS, device=dev)
        model.blocks[-1].capture = 64 if i == attn_at else 0
        v_c = model(ctx4, rays4, valid, x, tgt_rays, t)
        if i == attn_at:
            attn = model.blocks[-1].attn[0].cpu()  # (64 个目标 query, 全部 key)
        model.blocks[-1].capture = 0
        v_u = model(ctx4, rays4, torch.zeros_like(valid), x, tgt_rays, t)
        v = v_u + CFG * (v_c - v_u)
        frames.append({"t": t.item(), "x": x[0].clamp(-1, 1).cpu(), "x0": (x - t * v)[0].clamp(-1, 1).cpu()})
        x = x - v / N_STEPS
    frames.append({"t": 0.0, "x": x[0].clamp(-1, 1).cpu(), "x0": x[0].clamp(-1, 1).cpu()})
    return frames, attn


def prepare(scene_id: int, dev: str) -> dict:
    print("加载模型 ...")
    model = load_model(dev)
    scene = Scene(scene_id, dev)
    D = {}
    print("[1/4] 场景和相机 ...")
    D["world"] = world_cloud(scene, dev)
    D["ctx_R"], D["ctx_eye"] = cameras([0, 100], device=dev)
    D["tgt_R"], D["tgt_eye"] = cameras([50], device=dev)
    D["ctx"] = scene.shoot(D["ctx_R"], D["ctx_eye"])
    D["gt"] = scene.shoot(D["tgt_R"], D["tgt_eye"])[0]
    D["plucker"] = S.plucker(D["ctx_R"], D["ctx_eye"]).cpu()

    print("[2/4] 去噪 + 记录注意力 ...")
    attn_step = int(N_STEPS * 0.7)  # t = 0.3 时
    D["denoise"], attn = denoise(model, D["ctx"], D["ctx_R"], D["ctx_eye"], D["tgt_R"], D["tgt_eye"], dev,
                                 attn_at=attn_step)
    D["queries"] = []
    for py, px in pick_query_patches(D["gt"]):
        w = attn[py * 8 + px]
        corr = correspondence(D["gt"], D["tgt_R"][0], D["tgt_eye"][0], D["ctx_R"], D["ctx_eye"], py, px)
        D["queries"].append({"py": py, "px": px, "w": [w[c * 64:(c + 1) * 64].reshape(8, 8) for c in range(2)],
                             "corr": corr})

    print("[3/4] 自回归环绕（12 个视角）...")
    R0, e0 = cameras([0], device=dev)
    azims = [30 * (k + 1) for k in range(12)]
    Rp, ep = cameras(azims, device=dev)
    first, gt_orbit = scene.shoot(R0, e0)[0], scene.shoot(Rp, ep)
    imgs, Rs, es, orbit = [first], [R0[0]], [e0[0]], []
    for k in range(12):
        keep = [0] + list(range(max(1, len(imgs) - 3), len(imgs)))  # 真实照片 + 最近 3 张生成的
        frames, _ = denoise(model, torch.stack([imgs[i] for i in keep]), torch.stack([Rs[i] for i in keep]),
                            torch.stack([es[i] for i in keep]), Rp[k:k + 1], ep[k:k + 1], dev, seed=k)
        x = frames[-1]["x"].to(dev)
        orbit.append({"azim": azims[k], "img": x.cpu(), "window": [imgs[i].cpu() for i in keep],
                      "window_labels": [L("真实 0°", "real 0°") if i == 0 else L(f"生成 {azims[i - 1]}°", f"gen {azims[i - 1]}°")
                                        for i in keep],
                      "psnr": psnr(x, gt_orbit[k])})
        imgs.append(x)
        Rs.append(Rp[k])
        es.append(ep[k])
    D["first"], D["orbit"] = first.cpu(), orbit

    print("[4/4] 反投影成点云 ...")
    D["clouds"] = []
    for k in range(12):
        pts, cols = S.backproject(orbit[k]["img"], Rp[k].cpu(), ep[k].cpu())
        keep = (pts[:, :2].abs() < 2.3).all(1) & (pts[:, 2] < 2.5) & (pts[:, 2] > -0.3)
        D["clouds"].append((pts[keep].numpy(), cols[keep].numpy()))
    for k in ("ctx_R", "ctx_eye", "tgt_R", "tgt_eye", "ctx", "gt"):
        D[k] = D[k].cpu()
    print("准备完成。")
    return D


# ═══════════════════════════════ 绘图小工具 ═══════════════════════════════


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def frame_color(ax, color, lw=3):
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def frustum(ax, R, eye, color, scale=0.9, ls="-", lw=1.6):
    """画相机的视锥：相机中心 + 图像四个角的光线。"""
    eye = eye.numpy()
    corners = np.array([[0, 0], [S.IMG, 0], [S.IMG, S.IMG], [0, S.IMG]], dtype=float)
    d_cam = np.stack([(corners[:, 0] - S.IMG / 2) / S.FOCAL, (corners[:, 1] - S.IMG / 2) / S.FOCAL,
                      np.ones(4)], 1)
    pts = eye + scale * d_cam @ R.numpy()  # Rᵀ d：相机系 → 世界系
    for p in pts:
        ax.plot(*zip(eye, p), c=color, lw=lw, ls=ls)
    loop = np.vstack([pts, pts[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], c=color, lw=lw, ls=ls)
    ax.scatter(*eye, c=color, s=30)


def setup_3d(ax, lim=2.4, zmax=2.0):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((2 * lim, 2 * lim, zmax))
    ax.set_axis_off()


def exploded(img, gap=1):
    """把 32×32 的图拆成 8×8 个 patch，中间留缝，像一块块 token。"""
    im = rgb(img)
    out = np.ones((8 * (4 + gap) - gap, 8 * (4 + gap) - gap, 4))
    out[..., 3] = 0
    for y in range(8):
        for x in range(8):
            out[y * (4 + gap):y * (4 + gap) + 4, x * (4 + gap):x * (4 + gap) + 4, :3] = im[y * 4:y * 4 + 4, x * 4:x * 4 + 4]
            out[y * (4 + gap):y * (4 + gap) + 4, x * (4 + gap):x * (4 + gap) + 4, 3] = 1
    return out


def depth_img(x):
    return x[3].numpy()


# ═══════════════════════════════ 七个步骤 ═══════════════════════════════
# 每个 build(fig, D) 返回 (帧数, update(i), 帧间隔毫秒, 是否循环)


def step1_world(fig, D):
    ax3 = fig.add_axes([-0.04, 0.0, 0.66, 0.92], projection="3d")
    pts, cols = D["world"]
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=3, depthshade=False)
    for c in range(2):
        frustum(ax3, D["ctx_R"][c], D["ctx_eye"][c], CTX_COLORS[c])
    frustum(ax3, D["tgt_R"][0], D["tgt_eye"][0], TGT_COLOR, ls="--")
    setup_3d(ax3, lim=4.2, zmax=2.6)
    for c in range(2):
        ax = fig.add_axes([0.6 + c * 0.2, 0.5, 0.17, 0.3])
        ax.imshow(rgb(D["ctx"][c]))
        clean(ax)
        frame_color(ax, CTX_COLORS[c])
        ax.set_title(L(f"相机 {c + 1} 拍到的照片", f"Photo from camera {c + 1}"), fontsize=11, color=CTX_COLORS[c])
    ax = fig.add_axes([0.7, 0.13, 0.17, 0.3])
    ax.set_facecolor("#f0f0f0")
    ax.text(0.5, 0.5, "?", fontsize=60, ha="center", va="center", color=TGT_COLOR, transform=ax.transAxes)
    clean(ax)
    frame_color(ax, TGT_COLOR)
    ax.set_title(L("绿色虚线相机：我们想看到的新视角", "Green dashed camera: the new view we want"), fontsize=11, color=TGT_COLOR)

    def update(i):
        ax3.view_init(elev=28, azim=-60 + i * 2)

    return 180, update, 50, True


def step2_plucker(fig, D):
    ax3 = fig.add_axes([-0.04, 0.02, 0.58, 0.86], projection="3d")
    pts, cols = D["world"]
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1.5, alpha=0.35, depthshade=False)
    R, eye = D["ctx_R"][0], D["ctx_eye"][0]
    frustum(ax3, R, eye, CTX_COLORS[0])
    setup_3d(ax3, lim=4.2, zmax=2.6)
    ax3.view_init(elev=22, azim=-55)
    # 6×6 个像素的光线，颜色取该像素的颜色
    o, d = S.pixel_rays(R, eye)
    photo = rgb(D["ctx"][0])
    grid = [(v, u) for v in range(2, 32, 5) for u in range(2, 32, 5)]
    rays = []
    for v, u in grid:
        dn = d[v * S.IMG + u] / d[v * S.IMG + u].norm()
        end = eye + 5.5 * dn
        rays.append(ax3.plot(*zip(eye.numpy(), end.numpy()), c=photo[v, u], lw=1.2, alpha=0.9)[0])
        rays[-1].set_visible(False)
    fig.text(0.25, 0.1, L("红色相机：每个像素 → 一条从相机出发的 3D 光线（颜色 = 该像素的颜色）",
                          "Red camera: each pixel → a 3D ray from the camera (color = that pixel's color)"),
             fontsize=12, ha="center")

    # 伪彩色：把向量的 (x, y, z) 三个分量直接当成 (R, G, B)。两台相机用同一套映射，颜色才能互相比较
    pl = D["plucker"].permute(0, 2, 3, 1).numpy()  # (2, H, W, 6)
    d_show = (pl[..., :3] + 1) / 2  # 单位向量，分量在 [-1, 1]
    m_scale = np.abs(pl[..., 3:]).max()
    m_show = (pl[..., 3:] / m_scale + 1) / 2
    for c in range(2):
        for j, (name, im) in enumerate([(L("方向 d", "direction d"), d_show[c]), (L("矩 m = o × d", "moment m = o × d"), m_show[c])]):
            ax = fig.add_axes([0.55 + j * 0.22, 0.4 - c * 0.33, 0.19, 0.27])
            ax.imshow(np.clip(im, 0, 1))
            clean(ax)
            frame_color(ax, CTX_COLORS[c])
            ax.set_title(L(f"相机 {c + 1}：{name}", f"Camera {c + 1}: {name}"), fontsize=11, color=CTX_COLORS[c])
    fig.text(0.75, 0.855, "Plücker(u, v) = ( d ,  o × d ) ∈ ℝ⁶", fontsize=14, ha="center")
    fig.text(0.75, 0.82, L("右侧是伪彩色：把向量的 (x, y, z) 当成 (R, G, B) 显示，和照片内容无关。\n"
                           "只取决于相机在哪、朝哪看——所以是平滑渐变，看不到任何物体。",
                           "False color: the vector's (x, y, z) is shown as (R, G, B); it has nothing to do with the photo.\n"
                           "It depends only on where the camera is and where it looks, so it is a smooth gradient with no objects."),
             fontsize=10.5, ha="center", va="top", color="#b7791f")

    def update(i):
        for k, r in enumerate(rays):
            r.set_visible(k < i)

    return len(rays) + 1, update, 60, False


def step_moment(fig, D):
    """m = o × d：三个阶段——扫过像素 / o 换成线上别的点 / 反向看。"""
    R, eye = D["ctx_R"][0], D["ctx_eye"][0]
    o = eye.numpy()
    _, d_all = S.pixel_rays(R, eye)
    d_all = (d_all / d_all.norm(dim=1, keepdim=True)).numpy()

    ax3 = fig.add_axes([-0.04, 0.02, 0.58, 0.86], projection="3d")
    pts, cols = D["world"]
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1.5, alpha=0.25, depthshade=False)
    frustum(ax3, R, eye, CTX_COLORS[0])
    ax3.scatter([0], [0], [0], c="#222", s=60, zorder=5)
    setup_3d(ax3, lim=4.2, zmax=2.6)
    ax3.view_init(elev=24, azim=-58)
    para = Poly3DCollection([np.zeros((4, 3))], facecolor="#7F77DD", alpha=0.25, edgecolor="#7F77DD")
    ax3.add_collection3d(para)
    ray_ln, = ax3.plot([], [], [], c="#888780", lw=1.2, ls="--", label=L("光线（同一条直线）", "ray (one line)"))
    o_ln, = ax3.plot([], [], [], c="#378ADD", lw=2.5, marker="o", markevery=[1], label=L("o：原点 → 光线上一点（相机）", "o: origin → a point on the ray (camera)"))
    d_ln, = ax3.plot([], [], [], c="#D85A30", lw=3, marker=">", markevery=[1], label=L("d：光线方向（单位长度）", "d: ray direction (unit length)"))
    m_ln, = ax3.plot([], [], [], c="#534AB7", lw=3.5, marker="^", markevery=[1], label="m = o × d")
    h_ln, = ax3.plot([], [], [], c="#BA7517", lw=2, ls=":", label=L("原点到光线的距离 = |m|", "distance from origin to ray = |m|"))
    ax3.legend(loc="lower left", fontsize=9, frameon=False)
    fig.text(0.05, 0.84, L("黑点 = 世界原点（场景中心）", "Black dot = world origin (scene center)"), fontsize=10, color="#555")

    pl = D["plucker"].permute(0, 2, 3, 1).numpy()
    m_show = (pl[0, ..., 3:] / np.abs(pl[..., 3:]).max() + 1) / 2
    ax_m = fig.add_axes([0.56, 0.47, 0.17, 0.31])
    ax_m.imshow(np.clip(m_show, 0, 1))
    clean(ax_m)
    frame_color(ax_m, CTX_COLORS[0])
    ax_m.set_title(L("相机 1 的 m 图（伪彩色）", "Camera 1: m map (false color)"), fontsize=10, color=CTX_COLORS[0])
    mark = Rectangle((0, 0), 1.6, 1.6, fill=False, ec="yellow", lw=2.5)
    ax_m.add_patch(mark)
    phase = fig.text(0.76, 0.76, "", fontsize=13, weight="bold", va="top")
    info = fig.text(0.76, 0.69, "", fontsize=11, va="top", linespacing=1.7)
    fig.text(0.56, 0.4, L("m = o × d 编码光线“在哪里”：\n"
                          "· |m| = |o|·sinθ = 原点到光线的距离（= 紫色平行四边形面积）\n"
                          "· m 的方向垂直于“原点 + 光线”所在的平面\n"
                          "· o 沿 d 滑动 → m 不变：照片没有深度，本来就不该依赖取线上哪个点\n"
                          "· 反向看 d → −d  ⇒  m → −m\n"
                          "· d 管“朝哪看”，m 管“在哪”，合起来唯一确定一条光线",
                          "m = o × d encodes WHERE the ray is:\n"
                          "· |m| = |o|·sinθ = distance from origin to ray (= purple parallelogram area)\n"
                          "· m is perpendicular to the plane through the origin and the ray\n"
                          "· slide o along d → m unchanged: a photo has no depth,\n"
                          "   so the point we pick on the ray should not matter\n"
                          "· look the other way d → −d  ⇒  m → −m\n"
                          "· d says which way the ray points, m says where it is"),
             fontsize=10.5, va="top", linespacing=1.9)

    sweep = [(u, 16) for u in range(2, 31, 2)] + [(16, v) for v in range(2, 31, 2)]
    n_a, n_b, n_c = len(sweep), 36, 14
    slide_px = (6, 8)

    def set3(line, a, b):
        line.set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])

    def update(i):
        if i < n_a:
            (u, v), s, flip = sweep[i], 0.0, 1
            phase.set_text(L("阶段 1/3：扫过不同像素", "Phase 1/3: sweep pixels"))
        elif i < n_a + n_b:
            (u, v), flip = slide_px, 1
            s = 2.4 * math.sin((i - n_a) / n_b * 2 * math.pi)
            phase.set_text(L("阶段 2/3：把 o 换成光线上别的点", "Phase 2/3: slide o along ray"))
        else:
            (u, v), s, flip = slide_px, 0.0, -1
            phase.set_text(L("阶段 3/3：反向看（d → −d）", "Phase 3/3: reverse d → −d"))
        d = flip * d_all[v * S.IMG + u]
        p = o + s * d  # 光线上的一点
        m = np.cross(p, d)
        foot = p - np.dot(p, d) * d
        para.set_verts([np.array([[0, 0, 0], p, p + d, d])])
        set3(ray_ln, p - 6 * d, p + 6 * d)
        set3(o_ln, [0, 0, 0], p)
        set3(d_ln, p, p + 1.2 * d)
        set3(m_ln, [0, 0, 0], 0.8 * m)
        set3(h_ln, [0, 0, 0], foot)
        mark.set_xy((u - 0.8 - 0.5, v - 0.8 - 0.5))
        theta = math.degrees(math.acos(np.clip(np.dot(p, d) / np.linalg.norm(p), -1, 1)))
        info.set_text(L("像素", "pixel") + f" (u={u:2d}, v={v:2d})\n"
                      f"|o| = {np.linalg.norm(p):.2f}    θ = {theta:5.1f}°\n"
                      f"|m| = |o|·sinθ = {np.linalg.norm(m):.2f}\n"
                      f"m = ({m[0]:+.2f}, {m[1]:+.2f}, {m[2]:+.2f})")

    return n_a + n_b + n_c, update, 150, True


def step3_tokens(fig, D):
    ax = fig.add_axes([0.02, 0.32, 0.26, 0.48])
    ax.imshow(rgb(D["ctx"][0]))
    for k in range(1, 8):
        ax.axhline(k * 4 - 0.5, c="w", lw=1)
        ax.axvline(k * 4 - 0.5, c="w", lw=1)
    clean(ax)
    frame_color(ax, CTX_COLORS[0])
    ax.set_title(L("照片切成 8×8 个 patch（每块 4×4 像素）", "Photo cut into 8×8 patches (4×4 pixels each)"), fontsize=11)

    panels = [(exploded(D["ctx"][0]), L("上下文 1：64 个 token（干净）", "Context 1: 64 tokens (clean)"), CTX_COLORS[0]),
              (exploded(D["ctx"][1]), L("上下文 2：64 个 token（干净）", "Context 2: 64 tokens (clean)"), CTX_COLORS[1]),
              (exploded(D["denoise"][0]["x"]), L("目标：64 个 token（纯噪声）", "Target: 64 tokens (pure noise)"), TGT_COLOR)]
    ims = []
    for j, (im, title, color) in enumerate(panels):
        a = fig.add_axes([0.32 + j * 0.23, 0.36, 0.2, 0.4])
        ims.append((a.imshow(np.zeros_like(im)), im))
        a.set_xlim(-0.5, im.shape[1] - 0.5)
        a.set_ylim(im.shape[0] - 0.5, -0.5)
        clean(a)
        frame_color(a, color)
        a.set_title(title, fontsize=11, color=color)
    fig.text(0.62, 0.24, L("每个 token = W_img · patch  +  W_ray · Plücker  +  角色（上下文 / 目标）  +  位置",
                          "token = W_img · patch  +  W_ray · Plücker  +  role (context / target)  +  position"),
             fontsize=13, ha="center")
    fig.text(0.62, 0.18, L("3 × 64 = 192 个 token 排成一条序列，送进 8 层 Transformer（双向注意力）",
                          "3 × 64 = 192 tokens form one sequence → an 8-layer Transformer (bidirectional attention)"),
             fontsize=13, ha="center")

    def update(i):
        n = i * 4  # 每帧多显示 4 个 token
        for k, (artist, im) in enumerate(ims):
            show = np.array(im)
            m = min(max(n - k * 64, 0), 64)
            for idx in range(m, 64):
                y, x = divmod(idx, 8)
                show[y * 5:y * 5 + 4, x * 5:x * 5 + 4, 3] = 0.08
            artist.set_data(show)

    return 49, update, 60, False


def step4_attention(fig, D):
    qs = D["queries"]
    ax_t = fig.add_axes([0.03, 0.2, 0.3, 0.55])
    ax_t.imshow(rgb(D["gt"]))
    clean(ax_t)
    frame_color(ax_t, TGT_COLOR)
    ax_t.set_title(L("目标视角（显示真实图方便对照）", "Target view (ground truth shown for reference)"), fontsize=11, color=TGT_COLOR)
    box = Rectangle((0, 0), 4, 4, fill=False, ec="yellow", lw=3)
    ax_t.add_patch(box)
    heat, marks, axes = [], [], []
    for c in range(2):
        a = fig.add_axes([0.38 + c * 0.31, 0.2, 0.28, 0.55])
        a.imshow(rgb(D["ctx"][c]))
        heat.append(a.imshow(np.zeros((8, 8)), cmap="inferno", alpha=0.6, extent=(-0.5, 31.5, 31.5, -0.5),
                             interpolation="nearest", vmin=0))
        marks.append(a.plot([], [], "x", c="cyan", ms=16, mew=3)[0])
        clean(a)
        frame_color(a, CTX_COLORS[c])
        axes.append(a)
    hits = total = 0
    for q in qs:
        for c in range(2):
            if q["corr"][c] is not None:
                k = int(q["w"][c].argmax())
                ky, kx = divmod(k, 8)
                total += 1
                hits += math.hypot(kx * 4 + 2 - q["corr"][c][0], ky * 4 + 2 - q["corr"][c][1]) <= 6
    fig.text(0.5, 0.12, L(f"青色 × = 按真实 3D 几何算出的对应点。本场景 {hits}/{total} 次注意力峰值落在对应点一个 patch 以内"
                          f"——没人教过它几何，是训练自己学会的。",
                          f"Cyan × = true correspondence from 3D geometry. In this scene {hits}/{total} attention peaks land "
                          f"within one patch of it.\nNobody taught the model geometry; it learned this from training."),
             ha="center", va="top", fontsize=12)

    def update(i):
        q = qs[i % len(qs)]
        box.set_xy((q["px"] * 4 - 0.5, q["py"] * 4 - 0.5))
        for c in range(2):
            w = q["w"][c].numpy()
            heat[c].set_data(w)
            heat[c].set_clim(0, max(w.max(), 1e-6))
            corr = q["corr"][c]
            marks[c].set_data([corr[0] - 0.5] if corr else [], [corr[1] - 0.5] if corr else [])
            axes[c].set_title(L(f"相机 {c + 1}：黄框位置的注意力（占总注意力 {w.sum():.0%}）",
                                f"Camera {c + 1}: attention from the yellow box ({w.sum():.0%} of total)"), fontsize=11,
                              color=CTX_COLORS[c])

    return len(qs), update, 1800, True


def step5_denoise(fig, D):
    fr = D["denoise"]
    for c in range(2):
        a = fig.add_axes([0.02, 0.5 - c * 0.3, 0.13, 0.25])
        a.imshow(rgb(D["ctx"][c]))
        clean(a)
        frame_color(a, CTX_COLORS[c])
        a.set_title(L(f"上下文 {c + 1}", f"Context {c + 1}"), fontsize=10, color=CTX_COLORS[c])
    specs = [(L("x_t：模型此刻看到的", "x_t: what the model sees now"), lambda f: rgb(f["x"]), None),
             (L("x₀ 估计 = x_t − t·v（RGB）", "x₀ estimate = x_t − t·v (RGB)"), lambda f: rgb(f["x0"]), None),
             (L("x₀ 估计（深度）", "x₀ estimate (depth)"), lambda f: depth_img(f["x0"]), "magma")]
    arts = []
    for j, (title, fn, cmap) in enumerate(specs):
        a = fig.add_axes([0.19 + j * 0.2, 0.3, 0.18, 0.46])
        arts.append((a.imshow(fn(fr[0]), cmap=cmap, vmin=-1 if cmap else None, vmax=1 if cmap else None), fn))
        clean(a)
        a.set_title(title, fontsize=11)
    a = fig.add_axes([0.81, 0.4, 0.15, 0.27])
    a.imshow(rgb(D["gt"]))
    clean(a)
    frame_color(a, TGT_COLOR)
    a.set_title(L("真实答案（对照）", "Ground truth (reference)"), fontsize=10, color=TGT_COLOR)
    bar_ax = fig.add_axes([0.19, 0.2, 0.58, 0.04])
    bar = bar_ax.barh([0], [1.0], color="#c53030")[0]
    bar_ax.set_xlim(0, 1)
    bar_ax.set_yticks([])
    bar_ax.set_xlabel(L("噪声程度 t（1 = 纯噪声，0 = 干净）", "noise level t (1 = pure noise, 0 = clean)"), fontsize=10)
    t_text = fig.text(0.48, 0.8, "", ha="center", fontsize=14)

    def update(i):
        f = fr[i]
        for art, fn in arts:
            art.set_data(fn(f))
        bar.set_width(f["t"])
        t_text.set_text(L(f"第 {i}/{N_STEPS} 步   t = {f['t']:.2f}   每步 x ← x − v / {N_STEPS}",
                          f"step {i}/{N_STEPS}   t = {f['t']:.2f}   each step x ← x − v / {N_STEPS}"))

    return len(fr), update, 120, False


def step6_autoregressive(fig, D):
    orbit = D["orbit"]
    fig.text(0.22, 0.85, L("上下文窗口（送进 Transformer 的已知视角）", "Context window (known views fed to the Transformer)"),
             fontsize=12, ha="center")
    win_axes = [fig.add_axes([0.02 + j * 0.1, 0.63, 0.09, 0.17]) for j in range(4)]
    map_ax = fig.add_axes([0.04, 0.06, 0.32, 0.5])
    pts, cols = D["world"]
    map_ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=1, alpha=0.5)
    map_ax.set_xlim(-5.2, 5.2)
    map_ax.set_ylim(-5.2, 5.2)
    map_ax.set_aspect("equal")
    clean(map_ax)
    map_ax.set_title(L("俯视图：相机位置", "Top view: camera positions"), fontsize=11)
    ring = np.linspace(0, 2 * np.pi, 100)
    r_xy = 4.5 * math.cos(math.radians(25))
    map_ax.plot(r_xy * np.cos(ring), r_xy * np.sin(ring), c="#ccc", lw=1, ls=":")
    map_ax.scatter([r_xy], [0], c=CTX_COLORS[0], s=80, zorder=3, label=L("真实照片", "real photo"))
    cam_dots = map_ax.scatter([], [], c="#555", s=40, zorder=3)
    cur = map_ax.scatter([], [], c=TGT_COLOR, s=140, marker="*", zorder=4)
    map_ax.legend(loc="lower left", fontsize=9)
    gen_axes = [fig.add_axes([0.46 + (k % 4) * 0.13, 0.6 - (k // 4) * 0.26, 0.12, 0.2]) for k in range(12)]
    fig.text(0.715, 0.85, L("逐个生成的新视角（标题：方位角 与 真实图的 PSNR）",
                           "Views generated one by one (title: azimuth, PSNR vs. ground truth)"), fontsize=12, ha="center")

    def update(i):
        step = orbit[i]
        for j, a in enumerate(win_axes):
            a.clear()
            clean(a)
            if j < len(step["window"]):
                a.imshow(rgb(step["window"][j]))
                frame_color(a, CTX_COLORS[0] if j == 0 else "#888")
                a.set_title(step["window_labels"][j], fontsize=9)
            else:
                a.set_facecolor("#f5f5f5")
                a.set_title(L("（空）", "(empty)"), fontsize=9, color="#aaa")
        az = np.radians([o["azim"] for o in orbit[:i]])
        cam_dots.set_offsets(np.c_[r_xy * np.cos(az), r_xy * np.sin(az)] if i else np.empty((0, 2)))
        a_cur = math.radians(step["azim"])
        cur.set_offsets([[r_xy * math.cos(a_cur), r_xy * math.sin(a_cur)]])
        for k, a in enumerate(gen_axes):
            a.clear()
            clean(a)
            if k <= i:
                a.imshow(rgb(orbit[k]["img"]))
                frame_color(a, TGT_COLOR if k == i else "#bbb", lw=3 if k == i else 1)
                a.set_title(f"{orbit[k]['azim']}°   {orbit[k]['psnr']:.1f} dB", fontsize=10)
            else:
                a.set_facecolor("#f5f5f5")

    return len(orbit), update, 1000, False


def step7_pointcloud(fig, D):
    ax3 = fig.add_axes([-0.02, 0.08, 0.62, 0.8], projection="3d")
    setup_3d(ax3)
    count = fig.text(0.3, 0.08, "", fontsize=13, ha="center")
    scat = ax3.scatter([], [], [], s=3, depthshade=False)
    ax_img = fig.add_axes([0.63, 0.5, 0.16, 0.3])
    ax_dep = fig.add_axes([0.81, 0.5, 0.16, 0.3])
    ax_gt = fig.add_axes([0.62, 0.0, 0.38, 0.4], projection="3d")
    pts, cols = D["world"]
    ax_gt.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1, depthshade=False)
    setup_3d(ax_gt)
    ax_gt.view_init(elev=30, azim=-60)
    ax_gt.set_title(L("对照：真实场景", "Reference: true scene"), fontsize=10)
    for a in (ax_img, ax_dep):
        clean(a)
    clouds, orbit = D["clouds"], D["orbit"]
    sub = 5  # 每加一个视角，旋转 5 帧
    max_show = 4000  # 3D 散点重绘很慢，显示时最多画这么多点（计数仍按全部点）
    rng = np.random.default_rng(0)

    def update(i):
        k = min(i // sub, len(clouds) - 1)
        P = np.concatenate([c[0] for c in clouds[:k + 1]])
        C = np.concatenate([c[1] for c in clouds[:k + 1]])
        idx = rng.permutation(len(P))[:max_show] if len(P) > max_show else slice(None)
        scat._offsets3d = (P[idx, 0], P[idx, 1], P[idx, 2])
        scat.set_color(C[idx])
        ax3.view_init(elev=30, azim=-60 + i * 2.5)
        count.set_text(L(f"已融合 {k + 1}/12 个生成视角，共 {len(P)} 个点", f"Fused {k + 1}/12 generated views, {len(P)} points"))
        if i % sub == 0:
            ax_img.clear()
            ax_dep.clear()
            ax_img.imshow(rgb(orbit[k]["img"]))
            ax_dep.imshow(depth_img(orbit[k]["img"]), cmap="magma", vmin=-1, vmax=1)
            ax_img.set_title(L(f"刚加入：{orbit[k]['azim']}° 的 RGB", f"Just added: RGB at {orbit[k]['azim']}°"), fontsize=10)
            ax_dep.set_title(L("和它的深度", "and its depth"), fontsize=10)
            clean(ax_img)
            clean(ax_dep)

    return len(clouds) * sub, update, 90, False


STEPS = [
    (L("世界和相机", "World and cameras"),
     L("一个训练时从没见过的新场景。两台相机（红、蓝）各拍了一张照片，我们想知道绿色相机那个位置看到什么。",
       "A new scene never seen in training. Two cameras (red, blue) each took a photo. What would the green camera see?"),
     step1_world),
    (L("相机 → Plücker 射线", "Camera → Plücker rays"),
     L("模型不认识“相机参数”，只认识向量。每个像素反投影成一条 3D 光线，用 6 个数 (d, o×d) 表示——相机从此变成了和图像同样大小的一张“图”。",
       "The model only understands vectors. Each pixel becomes a 3D ray written as 6 numbers (d, o×d), so a camera turns into an 'image' of the same size."),
     step2_plucker),
    (L("矩 m = o × d：光线在哪里", "Moment m = o × d: where the ray is"),
     L("d 只说明光线朝哪看。m = o × d 补上“这条光线在哪”：离原点多远、在哪个平面里。和力矩 τ = r × F 是同一个东西。",
       "d only says which way the ray points. m = o × d adds where the ray is: how far from the origin, and in which plane. It is the same as torque τ = r × F."),
     step_moment),
    (L("切成 token", "Tokens"),
     L("和 LLM 把句子切成 token 一样：每张图切成 64 个 patch。上下文是干净的照片，目标一开始是纯噪声。",
       "Just as an LLM splits a sentence into tokens, each image becomes 64 patches. Context views are clean photos; the target starts as pure noise."),
     step3_tokens),
    (L("注意力：它在看哪里", "Attention: where it looks"),
     L("目标视角里黄框这个位置要画什么？模型去上下文照片里找。亮 = 注意力大（第 8 层，t = 0.3 时）。",
       "What goes in the yellow box of the target view? The model looks it up in the context photos. Bright = high attention (layer 8, at t = 0.3)."),
     step4_attention),
    (L("去噪：从纯噪声到新视角", "Denoising: from pure noise to a new view"),
     L("rectified flow：每一步预测方向 v，走一小步。右边的 x₀ 估计：一开始是模糊的“平均”，后来越来越确定。",
       "Rectified flow: predict the direction v and take a small step. The x₀ estimate starts as a blurry 'average' and becomes more and more certain."),
     step5_denoise),
    (L("自回归：绕场景一圈", "Autoregression: an orbit around the scene"),
     L("只给 1 张真实照片。每生成一张就放进上下文窗口（真实照片 + 最近 3 张生成的），再生成下一张——和 LLM 逐个 token 往后接完全一样。",
       "Only 1 real photo. Each new view joins the context window (real photo + last 3 generated) before the next one, just like an LLM appending tokens."),
     step6_autoregressive),
    (L("长出 3D", "Growing 3D"),
     L("3D 不是模型内部存的东西：每个生成视角的深度 → 反投影 X = R^T (D·K⁻¹[u,v,1]^T − t) → 融合成点云。",
       "3D is not stored inside the model: depth of each generated view → backproject X = R^T (D·K⁻¹[u,v,1]^T − t) → fuse into a point cloud."),
     step7_pointcloud),
]


def header(fig, k, title, caption):
    fig.text(0.02, 0.955, L(f"第 {k + 1}/{len(STEPS)} 步　{title}", f"Step {k + 1}/{len(STEPS)}   {title}"),
             fontsize=18, weight="bold")
    fig.text(0.02, 0.915, caption, fontsize=12, color="#444")


# ═══════════════════════════════ 播放器 ═══════════════════════════════


class Player:
    """整个会话只用一个一直在跑的 30ms 计时器，从不 stop/start。

    每一步的帧率靠"距上一帧过了多久"来控制；动画播完后 tick 什么也不做。
    原因：macOS 后端里，在计时器自己的回调中调用 timer.stop() 之后，
    之后的所有计时器都不再触发——动画会冻住。
    """

    BASE_MS = 30

    def __init__(self, fig, D):
        self.fig, self.D, self.k = fig, D, 0
        self.n, self.frame, self.loop, self.interval, self.finished = 1, 0, False, 100, True
        self.last = 0.0
        self.timer = fig.canvas.new_timer(interval=self.BASE_MS)
        self.timer.add_callback(self._tick)
        self.timer.start()
        fig.canvas.mpl_connect("key_press_event", self._key)

    def show(self, k):
        self.k = k
        self.fig.clf()
        title, caption, build = STEPS[k]
        header(self.fig, k, title, caption)
        self.fig.text(0.98, 0.015, L("→ / 空格 下一步　 ← 上一步　 R 重播　 Q 退出", "→ / Space next    ← back    R replay    Q quit"),
                      fontsize=9, color="#999", ha="right")
        self.n, self.update, self.interval, self.loop = build(self.fig, self.D)
        self.frame, self.finished = 0, self.n <= 1
        self.update(0)
        self.last = time.monotonic()
        self.fig.canvas.draw_idle()

    def advance(self):
        """前进一帧（不看时间）。返回 False 表示非循环动画已经播完。"""
        if self.finished:
            return False
        self.frame += 1
        if self.frame >= self.n:
            if not self.loop:
                self.frame, self.finished = self.n - 1, True
                return False
            self.frame = 0
        self.update(self.frame)
        self.fig.canvas.draw_idle()
        return True

    def _tick(self):
        now = time.monotonic()
        if not self.finished and (now - self.last) * 1000 >= self.interval:
            self.last = now
            self.advance()

    def _key(self, e):
        if e.key in ("right", " ") and self.k < len(STEPS) - 1:
            self.show(self.k + 1)
        elif e.key == "left" and self.k > 0:
            self.show(self.k - 1)
        elif e.key == "r":
            self.show(self.k)
        elif e.key == "q":
            plt.close(self.fig)
            # macOS 后端关掉窗口后，事件循环有时没被唤醒，程序就不退出（取决于有没有别的事件，
            # 比如鼠标移动）。这里没有要保存的状态，用后台线程兜底：0.5 秒后直接结束进程。
            sys.stdout.flush()
            threading.Timer(0.5, os._exit, args=(0,)).start()


def export(D):
    """无窗口模式：每一步存 3 帧（开始 / 中间 / 结束）。"""
    out = OUT / ("walkthrough_en" if LANG == "en" else "walkthrough")
    out.mkdir(parents=True, exist_ok=True)
    for k, (title, caption, build) in enumerate(STEPS):
        fig = plt.figure(figsize=(15, 8.4))
        header(fig, k, title, caption)
        n, update, _, _ = build(fig, D)
        for tag, i in [("a", 0), ("b", n // 2), ("c", n - 1)]:
            for j in range(i + 1) if build is step7_pointcloud else [i]:  # 点云那一步需要依次累积
                update(j)
            fig.savefig(out / f"step{k + 1}_{tag}.png", dpi=90)
        plt.close(fig)
    print(f"已导出到 {out}")


def export_gifs(D, dpi=72, max_mb=1.9):
    """每一步录成一个 GIF（适合贴到白板，白板单个 GIF 上限 2 MB）。

    结尾停 2 秒；超过 max_mb 时依次尝试：减少颜色 → 隔帧抽取 → 缩小尺寸。
    """
    from PIL import Image

    out = OUT / ("gifs_en" if LANG == "en" else "gifs")
    out.mkdir(parents=True, exist_ok=True)
    names = ["world", "plucker", "moment", "tokens", "attention", "denoise", "autoregressive", "pointcloud"]
    stride = {step1_world: 3, step3_tokens: 2}  # 180 帧的旋转、49 帧的 token 动画隔帧抽取
    attempts = [(256, 1, 1.0), (128, 1, 1.0), (128, 2, 1.0), (128, 2, 0.8), (96, 3, 0.7)]  # (颜色数, 额外抽帧, 缩放)

    def encode(path, frames, durations, loop, colors, skip, scale):
        fr, du = frames[::skip], [sum(durations[i:i + skip]) for i in range(0, len(durations), skip)]
        if not loop:
            du[-1] = max(du[-1], 2000)  # 结尾停一下再重播
        if scale != 1.0:
            w, h = fr[0].size
            fr = [f.resize((int(w * scale), int(h * scale)), Image.LANCZOS) for f in fr]
        q = [f.quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for f in fr]
        q[0].save(path, save_all=True, append_images=q[1:], duration=du, loop=0, optimize=True)
        return len(q), sum(du), fr[0].size

    for k, (title, caption, build) in enumerate(STEPS):
        fig = plt.figure(figsize=(15, 8.4), dpi=dpi)
        header(fig, k, title, caption)
        n, update, interval, loop = build(fig, D)
        step = stride.get(build, 1)
        frames, durations = [], []
        for i in range(0, n, step):
            update(i)
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
            durations.append(interval * step)
        plt.close(fig)
        path = out / f"step{k + 1}_{names[k]}.gif"
        for colors, skip, scale in attempts:
            nf, total, size = encode(path, frames, durations, loop and n > 1, colors, skip, scale)
            mb = path.stat().st_size / 1e6
            if mb <= max_mb:
                break
        note = "" if (colors, skip, scale) == attempts[0] else f"（已压缩：{colors} 色，每 {skip} 帧取 1，尺寸 ×{scale}）"
        print(f"  {path.name}: {nf} 帧，{total / 1000:.1f} 秒，{size[0]}×{size[1]}，{mb:.2f} MB{note}")
    print(f"已导出 GIF 到 {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=int, default=0)
    p.add_argument("--export", action="store_true")
    p.add_argument("--gif", action="store_true", help="把每一步录成 GIF，存到 out/gifs/")
    p.add_argument("--lang", choices=["zh", "en"], default="zh", help="画面语言：zh（默认）或 en")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    assert args.lang == LANG  # LANG 在导入时已经从命令行读出
    D = prepare(args.scene, args.device)
    if args.export:
        export(D)
        return
    if args.gif:
        export_gifs(D)
        return
    fig = plt.figure(figsize=(15, 8.4))
    fig.canvas.manager.set_window_title(L("Atlas-mini：一步一步看 world model 内部", "Atlas-mini: a step-by-step look inside a world model"))
    player = Player(fig, D)
    player.show(0)
    plt.show()


if __name__ == "__main__":
    main()
