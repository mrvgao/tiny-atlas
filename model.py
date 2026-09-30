"""
Atlas-mini：带相机位姿的 multimodal autoregressive diffusion transformer（缩小版）。

一次前向：
    [上下文视角 1..Vc 的 patch token（干净）] + [目标视角的 patch token（带噪声 x_t）]
    每个 token = 图像内容 + Plücker 射线（相机几何）+ 角色（上下文/目标）+ patch 位置
    时间 t 通过 adaLN（DiT 的做法）调制每一层
    输出：目标视角的速度场 v = ε - x₀（RGB 和深度一起）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from scenes import IMG


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(..., C, H, W) → (..., N, C·p·p)"""
    *lead, C, H, W = x.shape
    x = x.reshape(*lead, C, H // p, p, W // p, p)
    x = x.permute(*range(len(lead)), -4, -2, -5, -3, -1)  # (..., H/p, W/p, C, p, p)
    return x.reshape(*lead, (H // p) * (W // p), C * p * p)


def unpatchify(x: torch.Tensor, p: int, C: int) -> torch.Tensor:
    """(B, N, C·p·p) → (B, C, H, W)"""
    B, N, _ = x.shape
    h = w = int(math.sqrt(N))
    x = x.reshape(B, h, w, C, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C, h * p, w * p)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """和 Transformer 位置编码同样的 sin/cos，把标量 t ∈ [0,1] 变成向量。"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = (t * 1000)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    """标准 Transformer block，LayerNorm 的缩放/偏移由 t 决定（adaLN-Zero）。"""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)  # 初始时每个 block 都是恒等映射，训练更稳
        nn.init.zeros_(self.ada.bias)
        self.capture = 0  # >0 时记录最后 capture 个 query token 的注意力权重（只用于可视化）
        self.attn = None

    def forward(self, x, c, attn_mask):
        s1, g1, a1, s2, g2, a2 = self.ada(F.silu(c)).chunk(6, -1)
        B, L, D = x.shape
        q, k, v = self.qkv(modulate(self.norm1(x), s1, g1)).reshape(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if self.capture:
            w = q[:, :, -self.capture:] @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
            self.attn = w.masked_fill(~attn_mask, float("-inf")).softmax(-1).mean(1)  # 各头平均 (B, n_q, L)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # 双向注意力
        x = x + a1[:, None] * self.proj(h.transpose(1, 2).reshape(B, L, D))
        x = x + a2[:, None] * self.mlp(modulate(self.norm2(x), s2, g2))
        return x


class AtlasMini(nn.Module):
    def __init__(self, dim=256, depth=8, heads=4, patch=4, max_ctx=4):
        super().__init__()
        self.patch, self.max_ctx = patch, max_ctx
        n = (IMG // patch) ** 2
        self.img_in = nn.Linear(4 * patch * patch, dim)  # RGB-D patch
        self.ray_in = nn.Linear(6 * patch * patch, dim)  # Plücker patch —— 把 token 钉到 3D 空间
        self.role = nn.Embedding(2, dim)  # 0 = 干净的上下文，1 = 待去噪的目标
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)  # patch 在图里的位置
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, 4 * patch * patch)
        for m in (self.ada_out, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _tokens(self, rgbd, rays, role):
        """rgbd (B,V,4,H,W), rays (B,V,6,H,W) → (B, V·N, dim)"""
        B, V = rgbd.shape[:2]
        tok = self.img_in(patchify(rgbd, self.patch)) + self.ray_in(patchify(rays, self.patch))
        tok = tok + self.pos + self.role.weight[role]
        return tok.reshape(B, -1, tok.shape[-1])

    def forward(self, ctx_rgbd, ctx_rays, ctx_valid, x_t, tgt_rays, t):
        """
        ctx_rgbd  (B, Vc, 4, H, W)  干净的上下文视角（真实照片或之前生成的视角）
        ctx_rays  (B, Vc, 6, H, W)  上下文视角的 Plücker 射线
        ctx_valid (B, Vc) bool      哪些上下文槽位有效（张数可变 / CFG 丢弃）
        x_t       (B, 4, H, W)      加了噪声的目标视角
        tgt_rays  (B, 6, H, W)      目标相机的 Plücker 射线 —— "我要从这里看"
        t         (B,)              噪声程度，1 = 纯噪声，0 = 干净
        返回       (B, 4, H, W)      预测的速度 v ≈ ε - x₀
        """
        B, Vc = ctx_valid.shape
        n = (IMG // self.patch) ** 2
        tokens = torch.cat([self._tokens(ctx_rgbd, ctx_rays, 0),
                            self._tokens(x_t[:, None], tgt_rays[:, None], 1)], 1)
        key_valid = torch.cat([ctx_valid.repeat_interleave(n, 1),
                               torch.ones(B, n, dtype=torch.bool, device=t.device)], 1)
        attn_mask = key_valid[:, None, None, :]  # 无效的上下文 token 不能被看到
        c = self.t_mlp(timestep_embedding(t, 256))
        for blk in self.blocks:
            tokens = blk(tokens, c, attn_mask)
        shift, scale = self.ada_out(F.silu(c)).chunk(2, -1)
        h = self.out(modulate(self.norm_out(tokens[:, -n:]), shift, scale))  # 只取目标视角的 token
        return unpatchify(h, self.patch, 4)


@torch.no_grad()
def generate_view(model, ctx_rgbd, ctx_rays, ctx_valid, tgt_rays, steps=40, cfg=2.0,
                  noise=None, return_traj=False):
    """生成一张新视角：从纯噪声出发，沿 -v 做 steps 步欧拉积分（t: 1 → 0）。

    cfg > 1 时使用 classifier-free guidance：
        v = v_无条件 + cfg · (v_有条件 - v_无条件)
    "无条件"= 把所有上下文都遮掉，只保留目标相机。
    """
    B = tgt_rays.shape[0]
    x = noise if noise is not None else torch.randn(B, 4, IMG, IMG, device=tgt_rays.device)
    traj = [x]
    no_ctx = torch.zeros_like(ctx_valid)
    for i in range(steps):
        t = torch.full((B,), 1.0 - i / steps, device=x.device)
        v = model(ctx_rgbd, ctx_rays, ctx_valid, x, tgt_rays, t)
        if cfg != 1.0:
            v_u = model(ctx_rgbd, ctx_rays, no_ctx, x, tgt_rays, t)
            v = v_u + cfg * (v - v_u)
        x = x - v / steps
        traj.append(x)
    x = x.clamp(-1, 1)
    return (x, traj) if return_traj else x
