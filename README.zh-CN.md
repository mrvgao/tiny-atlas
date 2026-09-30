# tiny-atlas

[English](README.md) | **中文**

一个能在 Mac 上训练、端到端跑通的**迷你 world model**，用来讲清楚 World Labs [Atlas](https://www.worldlabs.ai/blog/atlas) 这类模型背后的原理。

**多视角数据 → Plücker 相机编码 → Transformer → rectified flow 去噪 → 自回归生成新视角 → 深度 → 点云**

![只给 1 张照片，自回归地绕场景生成一圈](assets/orbit.gif)

> 这是一个独立的教学实现，与 World Labs 无关。Atlas 没有公开论文和代码，这里的设计（尤其是用 Plücker 射线编码相机位姿）是按领域惯例做的合理推测，不代表 Atlas 的真实实现。

## 快速开始

```bash
pip install -r requirements.txt
python walkthrough.py     # 分步演示（推荐）：打开窗口，一步步展示模型内部
python demo.py            # 一次性生成 6 张演示图到 out/
```

仓库自带训练好的权重 `checkpoints/atlas_mini_ema.pt`（约 40 MB），克隆下来就能直接运行演示。有 Apple Silicon 时自动使用 Mac GPU（MPS），否则用 CPU。

想自己从头训练：

```bash
python train.py           # Mac GPU 约 1 小时（24000 步），每 2000 步在 checkpoints/previews/ 存一张预览图
```

## 分步演示（walkthrough）

```bash
python walkthrough.py            # 启动约 10 秒（预先把所有东西算好）
python walkthrough.py --scene 2  # 换一个训练时从没见过的测试场景
python walkthrough.py --gif      # 不开窗口，把每一步录成 GIF（每个 2 MB 以内），存到 out/gifs/
python walkthrough.py --export   # 不开窗口，把每一步的开始/中间/结束帧存成 PNG
python walkthrough.py --lang en  # 画面文字改成英文
```

按键：**→ / 空格** 下一步 · **←** 上一步 · **R** 重播当前动画 · **Q** 退出

| 步骤 | 画面 | 动画 |
|---|---|---|
| 1 世界和相机 | 场景点云 + 3 台相机的视锥 + 两张照片 + 一个"?" | 3D 视角缓慢旋转 |
| 2 Plücker 射线 | 从相机射出的光线扇面 + 两台相机的 d、m 图 | 光线一条条出现 |
| 3 矩 m = o × d | 3D 里的原点、o、d、平行四边形和 m，旁边是 m 图和实时数值 | 扫过像素 / o 沿光线滑动（m 不变）/ 反向看（m → −m） |
| 4 切成 token | 照片切成 8×8 格 → 3 组 token（上下文 ×2 + 噪声目标） | token 逐个亮起 |
| 5 注意力 | 目标视角里一个黄框 → 上下文照片上的注意力热图 + 真实对应点 × | 轮流换不同位置 |
| 6 去噪 | x_t、x₀ 估计（RGB 和深度）、t 进度条 | 40 步去噪逐帧播放 |
| 7 自回归 | 上下文窗口 + 俯视相机位置图 + 逐个生成的 12 个视角 | 每秒生成一张 |
| 8 长出 3D | 生成视角的深度逐个融合成点云 | 边增长边旋转 |

**第 5 步：注意力自己学会了 3D 对应**。目标视角里一个位置（黄框），在上下文照片里最关注哪里（亮处）；青色 × 是按真实 3D 几何算出的对应点。在测试场景上，最后三层的注意力峰值有 76%–84% 落在对应点一个 patch 以内（随机约为 12 像素远）——模型从没被教过几何。

![注意力](assets/step5_attention.gif)

**第 3 步：m = o × d 编码光线“在哪里”**。|m| 是原点到光线的距离；o 沿光线滑动时 m 不变（照片没有深度，编码本来就不该依赖取光线上的哪个点）；反向看时 m 变号。

![Plücker 矩](assets/step3_moment.gif)

**第 6 步：去噪**。下排是模型每一步对 x₀ 的估计：先是模糊的“平均”，再逐渐拿定主意。

![去噪](assets/step6_denoise.gif)

## 演示图（demo.py）

```bash
python demo.py                 # 输出 out/ 下 6 张图 + orbit.gif
python demo.py --scene 3       # 换测试场景
python demo.py --cfg 1.0       # 关掉 classifier-free guidance 对比
```

1. **看得越多，想象越少**：同一场景给 1/2/4 张上下文，新视角越来越接近真实（PSNR 上升）
2. **自回归环绕**：只给 1 张照片，每生成一张就接回上下文。转到背面时 PSNR 最低（纯靠想象），转回来又变好
3. **点云**：生成的深度反投影、多视角融合，和真实点云对比
4. **去噪过程**：x_t 和模型每一步对 x₀ 的估计
5. **分布而非答案**：同一张照片、不同噪声，照片里看不到的背面有不同的合理猜测；完全不给照片时是纯想象
6. **训练曲线**：每个 batch 都是全新的随机世界，模型没法背答案

![自回归环绕：PSNR 在正背面最低](assets/2_orbit.png)

![分布而非答案](assets/5_imagination.png)

## 代码结构

| 文件 | 内容 | 对应的概念 |
|---|---|---|
| `scenes.py` | 随机生成场景（地板 + 方块/球），光线求交渲染 RGB-D；`plucker()`；`backproject()` | 相机位姿的数学、深度反投影 |
| `model.py` | `AtlasMini`：patch token + Plücker + 角色 + 位置，adaLN 注入 t，双向注意力；`generate_view()` 含 CFG | 核心架构、空间上下文 |
| `train.py` | `flow_loss()`：抽 ε、抽 t、构造 x_t、目标 ε − x₀、MSE | rectified flow |
| `demo.py` | 6 张演示图 | 自回归、3D 输出 |
| `walkthrough.py` | 8 步交互演示 | 全流程 |

## 和真实 Atlas 的差别

| | tiny-atlas | Atlas |
|---|---|---|
| 规模 | 32×32，约 1000 万参数，训练 1 小时 | 1440p，参数量未公开 |
| VAE | 无，直接在像素上去噪 | latent diffusion |
| 数据 | 程序生成的方块世界 | 真实世界的图像、视频、位姿、深度 |
| 推理 | 每步重算整条序列 | KV cache 等 LLM 推理技术 |
| 位姿编码 | Plücker 射线 | 未公开（Plücker 是业界常见做法，属推测） |

## 训练中踩过的坑

- **噪声时间 t 的采样分布**：一开始用了 SD3 的 logit-normal，它几乎不训练 t < 0.1 的区间（只占 1.4%），生成结果满是残留噪点。改成均匀分布后问题消失。结论：noise schedule 要随分辨率调整，SD3 的设定是给高分辨率准备的。
- **EMA 要预热**：衰减 0.999 的 EMA 在训练初期几乎全是随机初始化的权重，早期 checkpoint 的生成质量反而比原始权重差。
- **macOS 上 matplotlib 动画会冻住**：在计时器自己的回调里调用 `timer.stop()` 之后，之后的所有计时器都不再触发。`walkthrough.py` 因此整个会话只用一个常驻计时器，按时间决定何时进下一帧。
