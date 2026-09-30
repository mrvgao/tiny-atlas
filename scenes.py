"""
程序生成的"迷你世界" + 光线求交渲染器。

每个场景：一块棋盘格地板 + 2~4 个随机的方块/球，一个平行光源（带阴影）+ 天空渐变。
每个视角输出 RGB + 深度，共 4 个通道，都归一化到 [-1, 1]。
训练时实时生成，每个 batch 都是全新的场景 —— 模型没法背下任何一个具体场景，
只能学到"世界一般长什么样"。这是 world model 与 3DGS（只拟合一个场景）的本质区别。

相机约定（和 Atlas 文档里的公式一致）：
    世界 → 相机：x_cam = R x_world + t，相机坐标系 x 向右、y 向下、z 朝前
    相机中心 o = -Rᵀ t，像素 (u, v) 的光线方向 d = Rᵀ K⁻¹ [u, v, 1]ᵀ
"""

import math

import torch
import torch.nn.functional as F

IMG = 32  # 图像分辨率
FOV_DEG = 50.0
FOCAL = (IMG / 2) / math.tan(math.radians(FOV_DEG / 2))
MAX_OBJ = 4
LIGHT = F.normalize(torch.tensor([0.5, 0.3, 1.0]), dim=0)
AMBIENT = 0.35
NEAR_DEPTH = 1.5  # 深度编码：反深度 NEAR_DEPTH / z，z ≤ NEAR_DEPTH 时为 1，无穷远（天空）为 0


# ─────────────────────────────── 场景与相机的随机采样 ───────────────────────────────


def _hsv_to_rgb(h, s, v):
    k = (torch.stack([5.0 + h * 6, 3.0 + h * 6, 1.0 + h * 6], -1)) % 6
    return v[..., None] - v[..., None] * s[..., None] * (torch.minimum(k, 4 - k).clamp(0, 1))


def sample_scenes(batch: int, gen: torch.Generator | None = None) -> dict:
    """随机生成 batch 个场景的参数（在 CPU 上采样，数据量很小）。"""
    r = lambda *shape: torch.rand(*shape, generator=gen)
    n_obj = torch.randint(2, MAX_OBJ + 1, (batch,), generator=gen)
    valid = torch.arange(MAX_OBJ)[None] < n_obj[:, None]  # (B, K)
    is_box = r(batch, MAX_OBJ) < 0.5
    # 尺寸：方块是三个半边长，球只用第一个数当半径
    half = 0.22 + 0.28 * r(batch, MAX_OBJ, 3)
    half[..., 2] = torch.where(is_box, 0.2 + 0.4 * r(batch, MAX_OBJ), half[..., 0])
    size = torch.where(is_box[..., None], half, half[..., :1].expand(-1, -1, 3))
    # 位置：四个物体分别放进四个象限（顺序打乱）再加抖动，避免互相穿插；底面贴地
    quadrants = torch.tensor([[-0.8, -0.8], [-0.8, 0.8], [0.8, -0.8], [0.8, 0.8]])
    perm = torch.argsort(r(batch, MAX_OBJ), dim=1)
    xy = quadrants[perm] + (r(batch, MAX_OBJ, 2) * 2 - 1) * 0.25
    center = torch.cat([xy, size[..., 2:3]], -1)
    angle = r(batch, MAX_OBJ) * math.pi / 2  # 方块绕竖直轴的朝向
    color = _hsv_to_rgb(r(batch, MAX_OBJ), 0.55 + 0.4 * r(batch, MAX_OBJ), 0.75 + 0.25 * r(batch, MAX_OBJ))
    return {"valid": valid, "is_box": is_box, "size": size, "center": center,
            "angle": angle, "color": color}


def look_at(eye: torch.Tensor) -> torch.Tensor:
    """eye (..., 3) → R (..., 3, 3)，相机看向原点上方一点。行向量依次是相机的 x(右), y(下), z(前)。"""
    target = torch.tensor([0.0, 0.0, 0.3], device=eye.device)
    fwd = F.normalize(target - eye, dim=-1)
    up = torch.tensor([0.0, 0.0, 1.0], device=eye.device).expand_as(fwd)
    right = F.normalize(torch.linalg.cross(fwd, up, dim=-1), dim=-1)
    down = torch.linalg.cross(fwd, right, dim=-1)
    return torch.stack([right, down, fwd], -2)


def orbit_eye(azim, elev, radius) -> torch.Tensor:
    """球坐标 → 相机位置。角度单位：弧度。"""
    return torch.stack([radius * torch.cos(elev) * torch.cos(azim),
                        radius * torch.cos(elev) * torch.sin(azim),
                        radius * torch.sin(elev)], -1)


def sample_cameras(batch: int, views: int, gen: torch.Generator | None = None):
    """每个场景随机 views 个相机：方位角任意，仰角 15°~45°，距离 3.5~5。返回 (R, eye)。"""
    azim = torch.rand(batch, views, generator=gen) * 2 * math.pi
    elev = torch.deg2rad(15 + 30 * torch.rand(batch, views, generator=gen))
    radius = 3.5 + 1.5 * torch.rand(batch, views, generator=gen)
    eye = orbit_eye(azim, elev, radius)
    return look_at(eye), eye


# ─────────────────────────────── 光线 ───────────────────────────────


def pixel_rays(R: torch.Tensor, eye: torch.Tensor):
    """每个像素的光线。R (..., 3, 3)，eye (..., 3) → o (..., 1, 3)，d (..., P, 3)。

    d 没有归一化：它在相机系里的 z 分量恰好是 1，所以求交得到的参数 t 就是 z-深度。
    """
    ys, xs = torch.meshgrid(torch.arange(IMG, device=R.device) + 0.5,
                            torch.arange(IMG, device=R.device) + 0.5, indexing="ij")
    d_cam = torch.stack([(xs - IMG / 2) / FOCAL, (ys - IMG / 2) / FOCAL, torch.ones_like(xs)], -1)
    d_cam = d_cam.reshape(-1, 3)  # (P, 3)
    d = d_cam @ R  # 等价于 Rᵀ d_cam：相机系方向 → 世界系方向
    return eye[..., None, :], d


def plucker(R: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """每个像素的 Plücker 射线 (d, o × d)，形状 (..., 6, IMG, IMG)。

    这是把相机"喂"给 Transformer 的方式：每个像素都知道自己那条光线在 3D 里的精确位置。
    """
    o, d = pixel_rays(R, eye)
    d = F.normalize(d, dim=-1)
    m = torch.linalg.cross(o.expand_as(d), d, dim=-1) / 4.0  # 除以 4 让数值量级和 d 接近
    ray = torch.cat([d, m], -1)  # (..., P, 6)
    return ray.transpose(-1, -2).reshape(*ray.shape[:-2], 6, IMG, IMG)


# ─────────────────────────────── 求交 ───────────────────────────────


def _hit_spheres(o, d, center, radius):
    """o (B,V,1,1,3), d (B,V,P,1,3), center (B,1,1,K,3), radius (B,1,1,K) → t (B,V,P,K)。"""
    oc = o - center
    a = (d * d).sum(-1)
    b = (oc * d).sum(-1)
    c = (oc * oc).sum(-1) - radius**2
    disc = b * b - a * c
    t = (-b - torch.sqrt(disc.clamp(min=0))) / a
    return torch.where((disc > 0) & (t > 1e-3), t, torch.full_like(t, math.inf))


def _to_box_frame(v, angle):
    """把向量绕 z 轴转 -angle，变到方块的局部坐标系。"""
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def _hit_boxes(o, d, center, half, angle):
    """返回 t (B,V,P,K) 和局部坐标系下的交点，用于算法向量。"""
    ol = _to_box_frame(o - center, angle)
    dl = _to_box_frame(d.expand(*d.shape[:-2], center.shape[-2], 3), angle)
    dl = torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
    t1, t2 = (-half - ol) / dl, (half - ol) / dl
    t_near = torch.minimum(t1, t2).amax(-1)
    t_far = torch.maximum(t1, t2).amin(-1)
    hit = (t_far > t_near) & (t_near > 1e-3)
    t = torch.where(hit, t_near, torch.full_like(t_near, math.inf))
    return t, ol + t_near[..., None] * dl


def _objects_t(o, d, sc):
    """所有物体的交点距离 (B,V,P,K)，无效物体为 inf。"""
    center = sc["center"][:, None, None]  # (B,1,1,K,3)
    size = sc["size"][:, None, None]
    angle = sc["angle"][:, None, None]
    t_s = _hit_spheres(o, d, center, size[..., 0])
    t_b, p_local = _hit_boxes(o, d, center, size, angle)
    t = torch.where(sc["is_box"][:, None, None], t_b, t_s)
    t = torch.where(sc["valid"][:, None, None], t, torch.full_like(t, math.inf))
    return t, p_local


# ─────────────────────────────── 渲染 ───────────────────────────────


@torch.no_grad()
def render(sc: dict, R: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """渲染 RGB-D。sc 是 sample_scenes 的输出，R (B,V,3,3)，eye (B,V,3) → (B,V,4,IMG,IMG)，范围 [-1,1]。"""
    dev = R.device
    sc = {k: v.to(dev) for k, v in sc.items()}
    B, V = R.shape[:2]
    o, d = pixel_rays(R, eye)  # (B,V,1,3), (B,V,P,3)
    o5, d5 = o[..., None, :], d[..., None, :]  # 多一个物体维度 K

    t_obj, p_local = _objects_t(o5, d5, sc)  # (B,V,P,K)
    t_min, k = t_obj.min(-1)  # 最近的物体
    t_floor = torch.where(d[..., 2] < -1e-6, -o[..., 2] / d[..., 2], torch.full_like(t_min, math.inf))
    hit_obj = t_min < t_floor
    t_hit = torch.minimum(t_min, t_floor)  # 这就是 z-深度
    hit_any = torch.isfinite(t_hit)
    p = o + t_hit.clamp(max=1e3)[..., None] * d  # 交点（世界系）

    # 法向量与颜色
    gather = lambda x: torch.gather(x[:, None, None].expand(-1, V, d.shape[2], *x.shape[1:]), 3,
                                    k[..., None, None].expand(-1, -1, -1, 1, *x.shape[2:]))[..., 0, :]
    center_k = gather(sc["center"])
    size_k = gather(sc["size"])
    color_k = gather(sc["color"])
    is_box_k = torch.gather(sc["is_box"][:, None, None].expand(-1, V, d.shape[2], -1), 3, k[..., None])[..., 0]
    angle_k = torch.gather(sc["angle"][:, None, None].expand(-1, V, d.shape[2], -1), 3, k[..., None])[..., 0]
    pl = torch.gather(p_local, 3, k[..., None, None].expand(-1, -1, -1, 1, 3))[..., 0, :]
    # 方块法向：局部坐标里离哪个面最近，就是哪个面的法向，再转回世界系
    rel = pl / size_k
    axis = rel.abs().argmax(-1, keepdim=True)
    n_local = torch.zeros_like(pl).scatter(-1, axis, torch.sign(torch.gather(rel, -1, axis)))
    c, s = torch.cos(angle_k), torch.sin(angle_k)
    n_box = torch.stack([c * n_local[..., 0] - s * n_local[..., 1],
                         s * n_local[..., 0] + c * n_local[..., 1], n_local[..., 2]], -1)
    n_sphere = F.normalize(p - center_k, dim=-1)
    n_obj = torch.where(is_box_k[..., None], n_box, n_sphere)
    n_floor = torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(p)
    normal = torch.where(hit_obj[..., None], n_obj, n_floor)

    # 大格子、低对比度的棋盘格：保留透视线索，又不会在 32×32 下产生严重混叠
    checker = ((torch.floor(p[..., 0] / 1.0) + torch.floor(p[..., 1] / 1.0)) % 2)[..., None]
    floor_col = (0.55 + 0.12 * checker) * torch.tensor([1.0, 0.97, 0.92], device=dev)
    albedo = torch.where(hit_obj[..., None], color_k, floor_col)

    # 阴影：从交点向光源发一条光线，看有没有被物体挡住
    light = LIGHT.to(dev)
    p_off = p + 1e-3 * normal
    t_shadow, _ = _objects_t(p_off[..., None, :], light.expand_as(p)[..., None, :], sc)
    lit = (t_shadow.min(-1).values == math.inf).float()
    diffuse = (normal * light).sum(-1).clamp(min=0) * lit
    shaded = albedo * (AMBIENT + (1 - AMBIENT) * diffuse)[..., None]

    # 天空：按光线仰角在地平线色和天顶色之间插值
    elev = F.normalize(d, dim=-1)[..., 2].clamp(0, 1)[..., None]
    sky = (1 - elev) * torch.tensor([0.85, 0.9, 0.95], device=dev) + elev * torch.tensor([0.35, 0.55, 0.85], device=dev)
    rgb = torch.where(hit_any[..., None], shaded, sky)

    inv_depth = torch.where(hit_any, (NEAR_DEPTH / t_hit).clamp(max=1.0), torch.zeros_like(t_hit))
    rgbd = torch.cat([rgb * 2 - 1, inv_depth[..., None] * 2 - 1], -1)  # (B,V,P,4)，映射到 [-1,1]
    return rgbd.transpose(-1, -2).reshape(B, V, 4, IMG, IMG)


def decode_depth(ch: torch.Tensor) -> torch.Tensor:
    """深度通道 [-1,1] → z-深度（天空为 inf）。"""
    inv = ((ch + 1) / 2).clamp(min=0)
    return torch.where(inv > 0.02, NEAR_DEPTH / inv.clamp(min=1e-3), torch.full_like(inv, math.inf))


def backproject(rgbd: torch.Tensor, R: torch.Tensor, eye: torch.Tensor):
    """单个视角的 RGB-D (4,H,W) → 世界系点云。对应文档里的 X = Rᵀ(D K⁻¹[u,v,1]ᵀ - t)。"""
    o, d = pixel_rays(R, eye)  # d 的相机系 z 分量为 1，所以 X = o + z·d
    z = decode_depth(rgbd[3]).reshape(-1)
    keep = torch.isfinite(z)
    pts = o + z[:, None].clamp(max=1e3) * d
    col = ((rgbd[:3] + 1) / 2).clamp(0, 1).reshape(3, -1).T
    return pts[keep], col[keep]
