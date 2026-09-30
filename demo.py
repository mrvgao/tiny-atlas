"""
课上演示：加载训练好的 Atlas-mini，在训练时从未见过的新场景上生成。

    python demo.py                 # 生成全部 6 张图 + 环绕 GIF，输出到 out/
    python demo.py --scene 7       # 换一个测试场景
    python demo.py --cfg 1.0       # 关掉 classifier-free guidance 对比

图 1  看得越多，想象越少：1 / 2 / 4 张上下文 → 新视角
图 2  自回归环绕：只给 1 张真实照片，逐个生成一圈新视角（RGB + 深度）
图 3  生成的 RGB-D → 反投影成点云，对比真实点云
图 4  去噪过程：x_t 和模型"心里"对 x₀ 的估计
图 5  同一张照片、不同噪声 → 没拍到的那一侧有不同的"想象"；完全不给照片 → 纯想象
图 6  训练曲线
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

import scenes as S
from model import AtlasMini, generate_view

HERE = Path(__file__).parent
OUT = HERE / "out"


# ─────────────────────────────── 工具 ───────────────────────────────


CKPT_FULL = HERE / "checkpoints" / "latest.pt"  # train.py 的完整输出（含优化器状态，约 160 MB，不进仓库）
CKPT_SLIM = HERE / "checkpoints" / "atlas_mini_ema.pt"  # 仓库自带：只有演示用的 EMA 权重（约 40 MB）


def load_model(device: str) -> AtlasMini:
    path = CKPT_FULL if CKPT_FULL.exists() else CKPT_SLIM
    ck = torch.load(path, map_location=device, weights_only=True)
    model = AtlasMini(**ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["ema"])  # 用 EMA 权重生成
    print(f"加载 checkpoint：训练了 {ck['step']} 步")
    return model


def cameras(azims_deg, elev_deg=25.0, radius=4.5, device="cpu"):
    """一组指定方位角的相机，返回 R (V,3,3), eye (V,3)。"""
    azim = torch.deg2rad(torch.tensor(azims_deg, dtype=torch.float32))
    elev = torch.full_like(azim, math.radians(elev_deg))
    eye = S.orbit_eye(azim, elev, torch.full_like(azim, radius))
    return S.look_at(eye).to(device), eye.to(device)


def rgb(x):
    return ((x[:3] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def depth(x):
    return x[3].cpu().numpy()


def clean(ax, ylabel=None):
    """隐藏刻度和边框，但保留 ylabel 作为行标签（ax.axis("off") 会连 ylabel 一起隐藏）。"""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def psnr(a, b):
    return (-10 * torch.log10((((a[:3] - b[:3]) / 2) ** 2).mean())).item()


class Scene:
    """一个测试场景：可以从任意相机"拍照"（得到真实 RGB-D）。"""

    def __init__(self, seed: int, device: str):
        self.params = S.sample_scenes(1, torch.Generator().manual_seed(10_000 + seed))
        self.device = device

    def shoot(self, R, eye):
        return S.render(self.params, R[None], eye[None])[0]  # (V,4,H,W)


def generate(model, ctx_rgbd, ctx_R, ctx_eye, tgt_R, tgt_eye, args, noise=None):
    """给定若干张上下文（真实或生成的）和若干个目标相机，批量生成目标视角。"""
    V_t, dev = tgt_R.shape[0], tgt_R.device
    n = ctx_rgbd.shape[0]
    pad = 4 - n
    ctx = torch.cat([ctx_rgbd, torch.zeros(pad, 4, S.IMG, S.IMG, device=dev)])
    rays = torch.cat([S.plucker(ctx_R, ctx_eye), torch.zeros(pad, 6, S.IMG, S.IMG, device=dev)])
    valid = torch.arange(4, device=dev) < n
    return generate_view(model, ctx[None].expand(V_t, -1, -1, -1, -1), rays[None].expand(V_t, -1, -1, -1, -1),
                         valid[None].expand(V_t, -1), S.plucker(tgt_R, tgt_eye),
                         steps=args.steps, cfg=args.cfg, noise=noise)


def autoregressive_orbit(model, first_rgbd, first_R, first_eye, R_path, eye_path, args):
    """沿相机轨迹逐个生成：上下文 = 真实的第一张 + 最近生成的 3 张（和 LLM 逐 token 往后接一样）。"""
    ctx_imgs, ctx_R, ctx_eye, outs = [first_rgbd], [first_R], [first_eye], []
    for k in range(R_path.shape[0]):
        keep = [0] + list(range(max(1, len(ctx_imgs) - 3), len(ctx_imgs)))
        x = generate(model, torch.stack([ctx_imgs[i] for i in keep]), torch.stack([ctx_R[i] for i in keep]),
                     torch.stack([ctx_eye[i] for i in keep]), R_path[k:k + 1], eye_path[k:k + 1], args)[0]
        outs.append(x)
        ctx_imgs.append(x)
        ctx_R.append(R_path[k])
        ctx_eye.append(eye_path[k])
    return torch.stack(outs)


# ─────────────────────────────── 图 ───────────────────────────────


def fig1_more_context(model, scene, args):
    ctx_R, ctx_eye = cameras([0, 90, 180, 270], device=args.device)
    tgt_R, tgt_eye = cameras([45, 135, 225], elev_deg=35, radius=4.2, device=args.device)
    ctx, gt = scene.shoot(ctx_R, ctx_eye), scene.shoot(tgt_R, tgt_eye)
    rows = [("ground truth", None)]
    for n in (1, 2, 4):
        torch.manual_seed(0)
        rows.append((f"{n} context view{'s' if n > 1 else ''}",
                     generate(model, ctx[:n], ctx_R[:n], ctx_eye[:n], tgt_R, tgt_eye, args)))
    fig, axes = plt.subplots(len(rows), 8, figsize=(11, 5.2),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.25, 1, 1, 1]})
    for r, (label, gen) in enumerate(rows):
        n_used = 4 if gen is None else int(label[0])
        for j in range(4):
            if j < n_used:
                axes[r, j].imshow(rgb(ctx[j]))
        for j in range(3):
            axes[r, 5 + j].imshow(rgb(gt[j] if gen is None else gen[j]))
        score = "" if gen is None else f"\n{np.mean([psnr(gen[j], gt[j]) for j in range(3)]):.1f} dB"
        for j in range(8):
            if j == 0:
                clean(axes[r, 0], label + score)
            elif (j < 4 and j >= n_used) or j == 4:
                axes[r, j].axis("off")
            else:
                clean(axes[r, j])
    axes[0, 1].set_title("context views (real photos + camera poses)", fontsize=10, loc="left")
    axes[0, 6].set_title("novel views", fontsize=10)
    fig.suptitle("1. The more it sees, the less it imagines", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "1_more_context.png", dpi=120)
    plt.close(fig)


def fig2_orbit(model, scene, args):
    n = 12
    azims = [30 * (k + 1) for k in range(n)]
    R0, e0 = cameras([0], device=args.device)
    Rp, ep = cameras(azims, device=args.device)
    first, gt = scene.shoot(R0, e0)[0], scene.shoot(Rp, ep)
    torch.manual_seed(0)
    gen = autoregressive_orbit(model, first, R0[0], e0[0], Rp, ep, args)

    fig = plt.figure(figsize=(15, 6.0))
    gs = fig.add_gridspec(4, n + 1, height_ratios=[1, 1, 1, 1.4])
    ax = fig.add_subplot(gs[0:2, 0])
    ax.imshow(rgb(first))
    ax.set_title("only real\nphoto (0°)", fontsize=9)
    ax.axis("off")
    for k in range(n):
        for r, (img, cmap) in enumerate([(rgb(gt[k]), None), (rgb(gen[k]), None),
                                         (depth(gen[k]), "magma")]):
            a = fig.add_subplot(gs[r, k + 1])
            a.imshow(img, cmap=cmap, vmin=-1 if cmap else None, vmax=1 if cmap else None)
            clean(a, ["ground truth", "generated", "gen. depth"][r] if k == 0 else None)
            if r == 0:
                a.set_title(f"{azims[k]}°", fontsize=9)
    a = fig.add_subplot(gs[3, 1:])
    a.plot(azims, [psnr(gen[k], gt[k]) for k in range(n)], "o-", c="#c53030")
    a.set_xlabel("camera azimuth (degrees away from the real photo)")
    a.set_ylabel("PSNR (dB)")
    a.set_xticks(azims)
    a.grid(alpha=0.3)
    fig.suptitle("2. Autoregressive orbit from ONE photo: context = real photo + last 3 generated views", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "2_orbit.png", dpi=120)
    plt.close(fig)
    return first, gt, gen, Rp, ep


def fig3_pointcloud(scene, gen, gt, Rp, ep):
    idx = range(0, len(gen), 2)
    clouds = {}
    for name, views in [("ground truth RGB-D", gt), ("generated RGB-D", gen)]:
        pts, cols = zip(*[S.backproject(views[k].cpu(), Rp[k].cpu(), ep[k].cpu()) for k in idx])
        pts, cols = torch.cat(pts), torch.cat(cols)
        keep = (pts.abs() < 2.5).all(1)  # 只看场景中心附近
        clouds[name] = (pts[keep].numpy(), cols[keep].numpy())
    fig = plt.figure(figsize=(12, 5.5))
    for c, (name, (pts, cols)) in enumerate(clouds.items()):
        for r, (elev, azim) in enumerate([(35, -60), (80, -90)]):
            ax = fig.add_subplot(1, 4, 2 * c + r + 1, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=2, depthshade=False)
            ax.view_init(elev, azim)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_zlim(0, 2)
            ax.set_box_aspect((5, 5, 2))
            ax.set_axis_off()
            ax.set_title(f"{name}\n({'side' if r == 0 else 'top'} view)", fontsize=10)
    fig.suptitle("3. Depth → point cloud:  X = Rᵀ(D·K⁻¹[u, v, 1]ᵀ − t),  6 views fused", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "3_pointcloud.png", dpi=120)
    plt.close(fig)


@torch.no_grad()
def fig4_denoising(model, scene, args):
    ctx_R, ctx_eye = cameras([0, 120], device=args.device)
    tgt_R, tgt_eye = cameras([60], device=args.device)
    ctx, gt = scene.shoot(ctx_R, ctx_eye), scene.shoot(tgt_R, tgt_eye)[0]
    dev = args.device
    ctx4 = torch.cat([ctx, torch.zeros(2, 4, S.IMG, S.IMG, device=dev)])[None]
    rays4 = torch.cat([S.plucker(ctx_R, ctx_eye), torch.zeros(2, 6, S.IMG, S.IMG, device=dev)])[None]
    valid = torch.tensor([[True, True, False, False]], device=dev)
    tgt_rays = S.plucker(tgt_R, tgt_eye)
    torch.manual_seed(3)
    x = torch.randn(1, 4, S.IMG, S.IMG, device=dev)
    show_at = {0, 8, 16, 24, 32, 39}
    xs, x0s, ts = [], [], []
    for i in range(args.steps):
        t = torch.full((1,), 1.0 - i / args.steps, device=dev)
        v_c = model(ctx4, rays4, valid, x, tgt_rays, t)
        v_u = model(ctx4, rays4, torch.zeros_like(valid), x, tgt_rays, t)
        v = v_u + args.cfg * (v_c - v_u)
        if i in show_at or i == args.steps - 1:
            xs.append(x[0].clamp(-1, 1))
            x0s.append((x - t * v)[0].clamp(-1, 1))  # 由 x_t = (1-t)x₀ + tε 推出 x₀ = x_t - t·v
            ts.append(t.item())
        x = x - v / args.steps
    fig, axes = plt.subplots(2, len(xs) + 1, figsize=(2.1 * (len(xs) + 1), 4.8))
    for j in range(len(xs)):
        axes[0, j].imshow(rgb(xs[j]))
        axes[0, j].set_title(f"t = {ts[j]:.2f}", fontsize=10)
        axes[1, j].imshow(rgb(x0s[j]))
    axes[0, -1].imshow(rgb(gt))
    axes[0, -1].set_title("ground truth", fontsize=10)
    axes[1, -1].imshow(rgb(x.clamp(-1, 1)[0]))
    axes[1, -1].set_xlabel("final sample", fontsize=10)
    for a in axes.flat:
        clean(a)
    axes[0, 0].set_ylabel("x_t\n(what it sees)", fontsize=9)
    axes[1, 0].set_ylabel("x₀ estimate\n= x_t − t·v", fontsize=9)
    fig.suptitle("4. Denoising one novel view (2 context photos): early guesses are blurry averages, then it commits",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "4_denoising.png", dpi=120)
    plt.close(fig)


def fig5_imagination(model, scene, args):
    R0, e0 = cameras([0], device=args.device)
    Rt, et = cameras([180, 180, 180, 180, 180], device=args.device)  # 正背面：照片里完全看不到
    first, gt = scene.shoot(R0, e0), scene.shoot(Rt[:1], et[:1])[0]
    torch.manual_seed(1)
    with_photo = generate(model, first, R0, e0, Rt, et, args)
    no_photo = generate(model, first[:0], R0[:0], e0[:0], Rt, et, args)  # 0 张上下文
    fig, axes = plt.subplots(2, 7, figsize=(13, 4.6))
    axes[0, 0].imshow(rgb(first[0]))
    axes[0, 0].set_title("given photo (0°)", fontsize=9)
    axes[0, 1].imshow(rgb(gt))
    axes[0, 1].set_title("truth at 180°", fontsize=9)
    for j in range(5):
        axes[0, j + 2].imshow(rgb(with_photo[j]))
        axes[0, j + 2].set_title(f"sample {j + 1}", fontsize=9)
        axes[1, j + 2].imshow(rgb(no_photo[j]))
    axes[1, 1].text(0.5, 0.5, "no photo at all\n(camera pose only)\n→ pure imagination", ha="center",
                    va="center", fontsize=10, transform=axes[1, 1].transAxes)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle("5. A distribution, not one answer: different noise → different guesses for the unseen side", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "5_imagination.png", dpi=120)
    plt.close(fig)


def fig6_training():
    hist = json.loads((HERE / "checkpoints" / "history.json").read_text())
    steps, loss = zip(*hist)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(steps, loss, c="#c53030")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("flow matching MSE")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("6. Training: every batch is a brand-new random world", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "6_training_curve.png", dpi=120)
    plt.close(fig)


def orbit_gif(first, gt, gen):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    axes[0].imshow(rgb(first))
    axes[0].set_title("real photo (0°)", fontsize=10)
    ims = [axes[1].imshow(rgb(gt[0])), axes[2].imshow(rgb(gen[0]))]
    title = axes[2].set_title("", fontsize=10)
    axes[1].set_title("ground truth", fontsize=10)
    for a in axes:
        a.axis("off")
    fig.tight_layout()

    def update(k):
        ims[0].set_data(rgb(gt[k]))
        ims[1].set_data(rgb(gen[k]))
        title.set_text(f"generated  {30 * (k + 1)}°")
        return ims

    FuncAnimation(fig, update, frames=len(gen), interval=400).save(OUT / "orbit.gif", writer=PillowWriter(fps=3))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=int, default=0, help="测试场景编号（训练时从未出现）")
    p.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance 强度，1.0 = 关闭")
    p.add_argument("--steps", type=int, default=40, help="去噪步数")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    model = load_model(args.device)
    scene = Scene(args.scene, args.device)
    print("图 1 ...")
    fig1_more_context(model, scene, args)
    print("图 2 ...")
    first, gt, gen, Rp, ep = fig2_orbit(model, scene, args)
    print("图 3 ...")
    fig3_pointcloud(scene, gen, gt, Rp, ep)
    print("图 4 ...")
    fig4_denoising(model, scene, args)
    print("图 5 ...")
    fig5_imagination(model, scene, args)
    fig6_training()
    orbit_gif(first, gt, gen)
    print(f"完成，输出在 {OUT}")


if __name__ == "__main__":
    main()
