"""
训练 Atlas-mini。课前跑一次（Mac GPU 上约 40 分钟），课上用 demo.py 加载演示。

    python train.py                      # 默认配置
    python train.py --steps 2000         # 快速冒烟测试
    python train.py --resume             # 从 checkpoints/latest.pt 继续

每一步：
    1. 实时生成 batch 个全新的随机场景，每个场景拍 5 张：前 4 张是上下文，第 5 张是目标
    2. 每个样本随机保留 1~4 张上下文（10% 的概率全部丢掉，为 CFG 做准备）
    3. rectified flow：x_t = (1-t)·x₀ + t·ε，让模型预测 ε - x₀，MSE
训练中每 2000 步保存 checkpoint，并在 checkpoints/previews/ 下输出一张预览图。
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import scenes as S
from model import AtlasMini, generate_view

CKPT_DIR = Path(__file__).parent / "checkpoints"


def make_batch(batch: int, gen: torch.Generator, device: str, views: int = 5):
    """一批训练数据：场景参数和相机在 CPU 上随机采样，渲染在 GPU 上做。"""
    sc = S.sample_scenes(batch, gen)
    R, eye = S.sample_cameras(batch, views, gen)
    R, eye = R.to(device), eye.to(device)
    return S.render(sc, R, eye), S.plucker(R, eye)


def flow_loss(model, rgbd, rays, gen_device: torch.Generator | None = None, p_drop: float = 0.1):
    """和 flow demo 里完全相同的 5 步，只是 x₀ 换成了一张 RGB-D 图，条件换成了带位姿的上下文。"""
    B, dev = rgbd.shape[0], rgbd.device
    n_ctx = torch.randint(1, 5, (B,), device=dev)
    valid = torch.arange(4, device=dev)[None] < n_ctx[:, None]
    valid &= ~(torch.rand(B, device=dev) < p_drop)[:, None]  # CFG：一部分样本看不到任何上下文

    x0 = rgbd[:, 4]
    eps = torch.randn_like(x0)  # ① 噪声
    # ② t ~ 均匀分布。SD3 用 logit-normal 偏重中等噪声，适合高分辨率；
    #    但在 32×32 上它几乎不训练 t<0.1 的区间（只占 1.4%），生成结果会残留噪点
    t = torch.rand(B, device=dev)
    x_t = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * eps  # ③ 混合
    target = eps - x0  # ④ 目标方向
    v = model(rgbd[:, :4], rays[:, :4], valid, x_t, rays[:, 4], t)
    return F.mse_loss(v, target)  # ⑤ MSE


@torch.no_grad()
def save_preview(ema, step: int, device: str, out: Path):
    gen = torch.Generator().manual_seed(1234)  # 固定的验证场景，训练中从未出现
    rgbd, rays = make_batch(4, gen, device)
    valid = torch.tensor([[True, True, False, False]] * 4, device=device)  # 给 2 张上下文
    pred = generate_view(ema, rgbd[:, :4], rays[:, :4], valid, rays[:, 4], steps=30)
    show = lambda x: ((x[:3] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu()
    fig, axes = plt.subplots(4, 5, figsize=(7.5, 6.2))
    for b in range(4):
        cols = [rgbd[b, 0], rgbd[b, 1], pred[b], rgbd[b, 4]]
        for j, im in enumerate(cols):
            axes[b, j].imshow(show(im))
        axes[b, 4].imshow(pred[b, 3].cpu(), cmap="magma", vmin=-1, vmax=1)
    for j, title in enumerate(["context 1", "context 2", "generated", "ground truth", "gen. depth"]):
        axes[0, j].set_title(title, fontsize=9)
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"step {step}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"step_{step:06d}.png", dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=24000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    dev = args.device
    (CKPT_DIR / "previews").mkdir(parents=True, exist_ok=True)
    cfg = {"dim": args.dim, "depth": args.depth, "heads": args.heads}
    model = AtlasMini(**cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    start, history = 0, []
    if args.resume and (CKPT_DIR / "latest.pt").exists():
        ck = torch.load(CKPT_DIR / "latest.pt", map_location=dev, weights_only=True)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start, history = ck["step"], ck["history"]
        print(f"从 step {start} 继续")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AtlasMini {n_params / 1e6:.1f}M 参数，设备 {dev}，batch {args.batch}，共 {args.steps} 步")

    def lr_at(step):  # 线性 warmup + cosine 衰减到 10%
        if step < 500:
            return args.lr * step / 500
        prog = (step - 500) / max(1, args.steps - 500)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    gen = torch.Generator().manual_seed(start)  # 续训时换一批场景
    t0, running = time.time(), []
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        rgbd, rays = make_batch(args.batch, gen, dev)
        loss = flow_loss(model, rgbd, rays)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():  # EMA 带预热：训练初期衰减小，避免早期 checkpoint 里全是随机初始权重
            decay = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        running.append(loss.item())

        if (step + 1) % 100 == 0:
            avg = sum(running) / len(running)
            history.append((step + 1, avg))
            running = []
            el = time.time() - t0
            rate = (step + 1 - start) / el
            eta = (args.steps - step - 1) / rate / 60
            print(f"step {step + 1:6d}  loss {avg:.4f}  lr {lr_at(step):.1e}  {rate:.1f} it/s  剩余 ~{eta:.0f} 分钟", flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            ck = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                  "step": step + 1, "history": history, "cfg": cfg}
            torch.save(ck, CKPT_DIR / "latest.pt")
            torch.save({k: ck[k] for k in ("ema", "cfg", "step")}, CKPT_DIR / "atlas_mini_ema.pt")  # 演示只需要这些
            save_preview(ema, step + 1, dev, CKPT_DIR / "previews")
            (CKPT_DIR / "history.json").write_text(json.dumps(history))
    print("训练完成：checkpoints/latest.pt")


if __name__ == "__main__":
    main()
