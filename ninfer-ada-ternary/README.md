---
license: Apache License 2.0
tasks:
- text-generation
frameworks:
- pytorch
libraries:
- ninfer
- gguf
language:
- zh
- en
tags:
- ninfer
- ternary
- 1.58-bit
- bonsai
- ada
- sm-89
- rtx-4080-super
- windows
- quantization
- hadamard
base_model: Qwen/Qwen3.8-27B
new_version: ''
datasets: []
metrics: []
---

# NInfer on Ada · 三元 Bonsai 2 27B 实战移植（RTX 4080 SUPER / Windows）

> ## ⚠️ 发布声明（请先读）
>
> ### 本仓**只发布「工具与方法」**，**不发布权重**。
>
> | | |
> |---|---|
> | ✅ **发布** | 技术文档 · 打包器与验证工具（`tools/`）· 引擎侧源码改动（`patches/`） |
> | ❌ **不发布** | **任何模型权重**，以及由权重派生的 `.ninfer` 制品 |
> | 🙏 **尊重原作** | 权重版权归 **PrismML 及其上游（Qwen 体系）** 所有，按其各自许可发布；我们只做引擎侧的适配与工具 |
> | ⬇️ **权重怎么拿** | **请前往官方作者处下载**（PrismML / Qwen 官方渠道），并遵守其许可条款；拿到权重后，用本仓的文档与工具自行生成制品 |
> | 📄 **许可** | 本仓代码为 **Apache License 2.0**（**不覆盖**权重）；权重许可亦不覆盖本仓代码。详见 [NOTICE.md](NOTICE.md) |
> | 🧱 **基线（照这个来）** | **`Ambolio/ninfer-4090-windows` @ `6eb70a07`（`v1.0.8` 线，Ada / Windows / sm_89）** —— `patches/` 就是针对它写的。**不要用 v1.2.0 或其它线**（目录结构与格式注册表都不同）。获取与覆盖步骤见 [docs/01](docs/01-4080S-接入-NInfer.md) §2 |

---

> 把 **NInfer** 推理引擎接到 **消费级 Ada 卡（RTX 4080 SUPER, sm_89, Windows）**，
> 并把 **Ternary Bonsai 2 27B**（1.58-bit 三元 + Hadamard 旋转基）移植进 NInfer 原生 `.ninfer`。
> 本仓是**技术成品展示 + 可复现文档 + 工具链**，不是一份权重分发。

---

## TL;DR

| 项 | 结果 |
|---|---|
| **端到端正确性** | ✅ 输出通顺；**PPL 6.445**（黄金参照 6.8196 ± 0.5365，同量级且略优） |
| **单流速度（MTP K=2）** | ✅ **96.7 – 130.8 t/s**（4/5 真实负载 ≥ 101 t/s） |
| 纯解码（T=1） | 62 – 64 t/s |
| 预填充（prefill） | 356 – 439 tok/s |
| 制品尺寸 | **7.74 GB**（文本部分 6.696 GiB，落在 5.5–7.2 GiB 红线内） |
| 20 题社区基准（MMLU/MMLU-Pro/ZebraLogic/AIME/HumanEval） | **19/20 = 95%**（同引擎全精度 27B 亦为 19/20，**打平**） |

**它做了什么**：给 NInfer 加了一种它原本没有的权重格式（三元 + 运行时 Hadamard 旋转），
把 **MTP 头拼进三元 artifact**（否则够不到 KPI），并让整条链在本机 32 GB 的 Ada 卡上跑通、达标。
**全社区此前没有第二家**把三元接进 NInfer。

---

## ★ 投机解码：MTP 头是**拼**进去的 + 接受率

三元 Bonsai **本身不带 MTP 头**（包内没有 NextN 张量）。我们把 **Qwen3.8-27B 自带的 MTP 头**（单层 NextN）
**按 NInfer 的 `mtp/*` 规格移植、拼进三元 artifact**——**这一步就是"负收益 → 正收益"的转折点**。

| 项 | 值 |
|---|---|
| 头来源 | Qwen3.8-27B 自带 NextN 层（与本体同族、**同一份**） |
| **拼接精度** | 重建头 vs artifact 自带头：**12/12 余弦 ≥ 0.99966**、**7 个 norm 逐字节相同** ⇒ 借用是**数值精确**的 |
| 能直接借的前提 | MTP 块**不做 Hadamard 旋转**（残差流在**原始基**）|
| 拼装内容 | 按 NInfer 规格搬 `blk.64.nextn.*`：`qkv 14336` / `gate_up 34816` 融合 + `W8G32` 量化 |
| **K=2 接受率** | **≈ 59.4%**（**强依赖任务**：英文代码 **40.8%** → 中文短答 **80%**）|
| **tok/轮** | **2.19**（实测 **1.82 ~ 2.38**，±27%）|
| **窗口** | **N=2 最优**；N=3/4/5 接受率**崩**到 **40.9 / 33.2 / 22.9%**（N≥6 本线不支持）|
| 长度效应 | 说明文 71 token 时接受 **66.7%** → 120 token 掉到 **54.8%** |
| **净效果** | 纯解码 **62 → 96.7~130.8 t/s**（**KPI 就是靠它**）|

> ⚠️ 划重点：**没有这个拼进来的 MTP 头，就只有 62 t/s 的裸解码，够不到 100+。**
> 已否掉的替代：`--lm-head-draft`（优化提案头）**更差**——说明文接受率 63.5% → **55.0%**。

---

## 本仓包含什么

```
README.md                      ← 本文（模型卡片）
NOTICE.md                      ← 第三方来源与许可声明
LICENSE                        ← Apache License 2.0
docs/
  01-4080S-接入-NInfer.md       ← 让 NInfer 在 Ada/Windows 上跑起来
  02-微调模型转-NInfer.md       ← 把普通（微调）模型转成 .ninfer
  03-三元模型转-NInfer.md       ← 把三元模型移植进 NInfer（本项目主线）
  04-工程实录-坑与方法学.md      ← 踩过的坑与验证方法学（最值钱的部分）
tools/
  pack.py                      ← GGUF → .ninfer 三元打包器
  MAPPING.json                 ← 逐张量映射规格（权威）
  verify/                      ← 4 个装配/解包验证脚本 + 旋转/GEMM oracle
patches/
  changed-files/               ← 引擎侧源码改动（与上游 fork 同结构，覆盖即可）
  README-改动说明.md            ← 改动清单 + 重建 + 必须做的验证
```

---

## 三份技术文档（核心交付）

### ① [4080S 接入 NInfer](docs/01-4080S-接入-NInfer.md)
消费级 Ada（sm_89）+ Windows 上把 NInfer 编出来、跑起来：环境、构建、运行、档位、坑。

### ② [微调模型 → .ninfer](docs/02-微调模型转-NInfer.md)
把任意**受支持架构**的 HF safetensors 微调模型转成 NInfer 原生 `.ninfer`（用引擎自带的
`tools/convert/` 配方框架）。**零引擎改动**。

### ③ [三元模型 → .ninfer](docs/03-三元模型转-NInfer.md)
把**三元 + Hadamard 旋转基**的模型移植进 NInfer：新增格式、旋转内核、fused 家族接线、
GEMV/mma 两条快路径、打包器。**本项目的主体工作**。

---

## 数据从哪来（诚实交代）

- 速度/正确性数字全部来自**引擎自报 timings + PPL 实测**，不是估算；
- 每一步判据都写在对应文档里（"正确应看到什么、错误会看到什么"）；
- 关键结论都带**负控**与**独立旁证**（详见 ④）。

---

## 许可与来源

本项目为 **NInfer（Apache-2.0）** 的派生作品，并消费 **PrismML 的 Ternary Bonsai 2 27B** 权重。
详见 [NOTICE.md](NOTICE.md)。

- 引擎与工具：Apache License 2.0
- 权重（不在本仓分发）：来自 PrismML / Qwen 体系，遵循其各自许可
