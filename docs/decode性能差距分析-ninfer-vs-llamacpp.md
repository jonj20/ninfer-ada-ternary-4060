# PTQ1_0 decode 性能差距分析：ninfer vs llama.cpp-prism（2026-09-25）

> 同机 RTX 4060 8G、同一 PTQ1_0 GGUF 文件：llama.cpp-prism **25 tok/s** vs ninfer **8.33 tok/s**，差距 ×3.0。
> 本文档基于对两个代码库的深度对比，定位差距根因并给出优化方向。

---

## 1. 测试基线

| 项 | 值 |
|---|---|
| 硬件 | RTX 4060 8G，sm_89，24 SM，~272 GB/s |
| 模型 | Ternary Bonsai 2 27B，PTQ1_0（1.75 bpw），权重 5.52 GiB |
| 架构 | 64 层 hybrid（16 full attention + 48 GDN），hidden=5120 |
| ninfer decode | **8.33 tok/s**（greedy / rk4v4-e8 / ctx2048 稳态） |
| llama.cpp decode | **25 tok/s**（同文件同卡） |
| 理论带宽上限 | ~49 tok/s（5.52 GiB ÷ 272 GB/s），当前只用了 17% |

---

## 2. 核心结论

**差距不主要在 GEMV 点积指令（DP4A vs FMA），而在工程结构：kernel 数量、融合程度、SM 策略。**

DP4A 指令替换只能带来约 ×1.3-1.5 的改善（8.33 → ~11-13 tok/s），无法收窄到 25 tok/s。

---

## 3. 结构性差异对照（按影响排序）

### 3.1 每 token kernel 数量：ninfer ~1200-1400 vs llama.cpp 远少

ninfer 每个 decode token 执行的 kernel 分布：

| 类别 | 数量/步 | 说明 |
|---|---|---|
| PTQ1 GEMV | **400** | 16 full × 7 + 48 GDN × 6 |
| Hadamard 旋转 | **~200** | 每个折叠三值 GEMM 前独立 launch |
| rmsnorm | ~200 | 每层 3+ 次 |
| residual_add / silu_mul | ~200 | 不与 GEMV 融合 |
| attention（partial+reduce） | 16 × 2-3 | SmallT 两段式 |
| rope / sigmoid_mul / scatter / sample 等 | ~50 | |
| **合计** | **~1200-1400** | 全部包在一张 CUDA Graph 内 |

llama.cpp 通过算子融合（RMS_NORM+ROPE、gate+up+SwiGLU、residual+norm 等）显著减少 kernel 数。

**出处：** ninfer `text_context_impl.h` L799-972（attn_mix/mlp_tail）；llama.cpp `ggml-cuda.cu` L4089-4123（fusion pass）

### 3.2 Hadamard 旋转是独立小 kernel（~200 次/步）

| 项 | ninfer | llama.cpp |
|---|---|---|
| 实现 | 独立 kernel + 独立 scratch buffer | 与 matmul 调度耦合更紧 |
| grid | K=5120 → **grid=1**（1 CTA, 256 线程） | 融合进相邻 kernel |
| 每次流量 | ~40 KB（K=5120）/ ~136 KB（K=17408） | 无独立流量 |
| 频率 | ~200 次/token | 无独立 launch |

grid=1-3 CTA 的 kernel 是纯 latency 贡献：launch 固定成本 + 极小并行度。200 次累计是显著开销。

**出处：** ninfer `ternary_rotation.cu` L50-54, L85-86；`linear_add.cpp` L191-196

### 3.3 linear 层三段式不融合

ninfer 每个折叠三值线性层 = 3 个独立 kernel：

```
旋转(K) → GEMV(N×K) → residual_add / silu_mul
```

llama.cpp 的对应路径融合更充分：

```
RMS_NORM+MUL+ROPE+SET_ROWS 五合一（ggml-cuda.cu L4089-4107）
gate+up+SwiGLU 单 kernel（mmvq.cu L730-739, has_gate 融合）
residual+norm 融合（norm.cu L157-182 add_rms_norm_f32）
```

**影响：** 每层多 2-3 个 kernel launch + 额外的 scratch 读写。

**出处：** ninfer `linear_add.cpp` L187-197；`linear_swiglu.cpp` L120-132；llama.cpp `ggml-cuda.cu` L3696-3731

### 3.4 SM 数写死 4090（kTargetSmCount=128）

| 项 | 值 | 影响 |
|---|---|---|
| ninfer `kTargetSmCount` | **128**（写死为 4090） | GQA attention split 策略按 128 SM 设计 |
| 4060 实际 SM | **24** | split 数严重超配，每个 split 工作量不足 |
| I8 attention kMax | `128/KVHeads = 32` | 4060 上 32 splits × 4 heads = 128 CTA，但只有 24 SM |

**影响：** GQA attention 的 split-KV 策略在 4060 上严重欠并行，大量 CTA 排队等 SM。

**出处：** ninfer `src/core/device.h` L22-24；`gqa_attention_decode.cuh` L82-95, L112

### 3.5 GEMV 内核架构差异

| 维度 | ninfer | llama.cpp MMVQ |
|---|---|---|
| 并行映射 | 1 warp = 1 输出行 | 1 CTA = 4 输出行（small_k） |
| block | 256 线程（8 warp） | 128 线程（4 warp） |
| grid（N=5120） | 640 CTA | 1280 CTA（4 行/CTA） |
| 归约 | warp shuffle（无 `__syncthreads`） | shared memory + warp shuffle |
| 激活格式 | bf16 原生（无量化 kernel） | F32 → Q8_1（每次 mul_mat 多一次 `quantize_q8_1` launch） |
| 点积指令 | `fmaf` 链（4 条浮点指令） | `__dp4a`（1 条整数指令） |
| K 循环 | 顺序扫行内全部 group | VDR=4，每线程处理 1 个 128 block |

**关键：** ninfer 省掉了 Q8_1 量化 launch，但 bf16 激活带宽是 int8 的 2 倍；llama.cpp 多一次量化 launch 但激活带宽减半。两者是不同的工程取舍。

**出处：** ninfer `ternary_rowsplit_gemv.cuh` L39, L51, L116-291；llama.cpp `mmvq.cu` L406-555, L920-930

### 3.6 权重布局差异

| 维度 | ninfer | llama.cpp |
|---|---|---|
| 布局 | SoA 三平面（base/high/scale 各自连续） | AoS（qs/qh/d 交织在 28B block 内） |
| 组大小 | 128 权重 = 24+2+2 = 28 B | 同 |
| 平面对齐 | 256 B | 无额外对齐 |
| scale 访问 | 独立平面连续 2B 读 | 从 28B block 尾部取 2B |
| 行尾 padding | K pad 到 128 | `MATRIX_ROW_PADDING=512`（仅 cuBLAS 路径） |

两者总字节数相同（28 B/组），布局差异对 L2 命中率有细微影响但非主要瓶颈。

**出处：** ninfer `ternary_rowsplit_storage.cuh` L13-37；llama.cpp `ggml-common.h` L214-220

---

## 4. 优化方向（按预期收益排序）

### 方向 1：旋转与 GEMV 融合（预期收益最大）

**问题：** ~200 次/步的独立旋转 kernel，grid=1-3 CTA，纯 latency。

**方案：**
- 将旋转作为 GEMV kernel 的 epilogue/producer：GEMV 读激活时先做旋转再进点积
- 或将旋转合并到 rmsnorm 的 epilogue（norm → rotate → GEMV 两段式）
- 需要 GEMV kernel 读旋转后激活（从 scratch 改为在线计算）

**预期：** 减少 ~200 个 kernel/步，消除旋转的独立 launch + scratch 读写。

### 方向 2：residual_add / silu_mul 融合进 GEMV epilogue

**问题：** 每个 linear 层的 residual_add 或 silu_mul 是独立 kernel。

**方案：**
- GEMV 输出时直接累加 residual（`out[warp] += residual[warp]`）
- silu_mul 需要两个 GEMV 输出（gate + up），可在 gate_up GEMV 内融合 SwiGLU

**预期：** 减少 ~200 个 kernel/步 + 消除中间 scratch。

### 方向 3：修复 kTargetSmCount（最简单，立即生效）

**问题：** `kTargetSmCount=128` 写死为 4090，4060 只有 24 SM。

**方案：**
- 改为运行时查询 `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)`
- 或至少为 4060 加一个 24 SM 的编译分支

**预期：** GQA attention split 策略立即适配 4060，短/中上下文 attention 并行度提升。

**出处：** `src/core/device.h` L22-24；`gqa_attention_decode.cuh` L82-95

### 方向 4：激活 Q8_1 量化（需架构改动）

**问题：** bf16 激活带宽是 int8 的 2 倍。

**方案：**
- 每层 linear 输入前加 `quantize_q8_1` kernel（或融合进 norm epilogue）
- GEMV 改读 Q8_1 int8 + `__dp4a` 点积
- 需改 `validate_linear_semantics` 的 BF16 约束

**预期：** 激活带宽减半 + DP4A 指令效率，综合 ×1.5-2。但引入额外量化 kernel（可融合进 norm）。

### 方向 5：DP4A 指令替换（收益有限，之前已实现后回退）

**问题：** `fmaf` 链 4 条指令 vs `__dp4a` 1 条指令。

**预期：** 仅 ×1.3-1.5，不是主要瓶颈。如果方向 1-3 实施后仍有差距再考虑。

---

## 5. 建议验证步骤

1. **ncu profiling**：抓 ninfer decode 的 kernel 时间分布，确认 GEMV / 旋转 / attention / 其他各占多少
2. **kTargetSmCount 修复**：最小改动，立即测 GQA attention 改善
3. **旋转融合**：最大预期收益，需要改 GEMV kernel 的输入路径
4. **逐项 A/B**：每项改动后跑 17×23 冒烟 + tok/s 对比

---

## 6. 关键文件索引

### ninfer

| 主题 | 路径 |
|---|---|
| GEMV 分派 | `src/ops/linear/ternary/ternary_rowsplit_gemm.cu` |
| GEMV kernel | `src/ops/linear/ternary/ternary_rowsplit_gemv.cuh` |
| 旋转 | `src/ops/linear/ternary/ternary_rotation.{cu,cpp}` |
| linear_add（不融合） | `src/ops/linear/linear_add.cpp` L187-197 |
| linear_swiglu（不融合） | `src/ops/linear/linear_swiglu.cpp` L120-132 |
| 层调度 | `src/targets/qwen3_6/impl/runtime/text_context_impl.h` L799-972 |
| SM 数常量 | `src/core/device.h` L22-24 |
| GQA split 策略 | `src/ops/kernel/gqa_attention_decode.cuh` L82-95 |
| CUDA Graph | `src/core/decode_graph.cpp` L58-79 |
| 权重布局 | `src/ops/linear/ternary/ternary_rowsplit_storage.cuh` |

### llama.cpp-prism

| 主题 | 路径 |
|---|---|
| MMVQ kernel | `ggml/src/ggml-cuda/mmvq.cu` L557-848 |
| PTQ1_0 vec_dot | `ggml/src/ggml-cuda/vecdotq.cuh` L808-891 |
| Q8_1 量化 | `ggml/src/ggml-cuda/quantize.cu` L54-102, L638-653 |
| 算子融合 | `ggml/src/ggml-cuda/ggml-cuda.cu` L4089-4123 |
| gate+up 融合 | `ggml/src/ggml-cuda/mmvq.cu` L730-739 |
| CUDA Graph | `ggml/src/ggml-cuda/ggml-cuda.cu` L4530-4587 |
| 权重结构 | `ggml/src/ggml-common.h` L214-220 |
| MMVQ 选路 | `ggml/src/ggml-cuda/mmvq.cu` L293-382 |
