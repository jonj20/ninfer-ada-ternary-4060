# PTQ1_0 解码 GEMV 检查

`ternary_rowsplit_gemv.cuh` 里两个解码内核的**正确性 / 带宽 / 精度**检查，以及解码阶段的
GPU 时间预算拆解。背景与实测结论见 [docs/4060-开发跟踪.md](../../../docs/4060-开发跟踪.md) §9.1。

这些脚本不是一次性的：改解码内核时用它们守住回归。第一项是**改完先跑**的门禁。

## 快速开始

```bash
# 全部（约 2 分钟）
tools/verify/ternary_gemv/run_checks.sh

# 改完内核先跑这个（秒级）
tools/verify/ternary_gemv/run_checks.sh reference

# 单项
tools/verify/ternary_gemv/run_checks.sh bandwidth   # 带宽，多形状
tools/verify/ternary_gemv/run_checks.sh pattern     # 访存归因
tools/verify/ternary_gemv/run_checks.sh accuracy    # int8 激活精度
```

环境变量：`NINFER_ROOT`（默认本仓根）、`NINFER_ARCH`（默认 89）、`NVCC`、`WORK_DIR`
（默认 `out/ternary-gemv`）。需要一块 GPU。

## 各项在测什么

| 脚本 | 作用 | 判据 |
|---|---|---|
| `gemv_reference_check.cu` | 内核 vs CPU 双精度参考 | 每行相对误差 ≤ 2%（实测 0.0038，bf16 舍入量级）|
| `gemv_bandwidth.cu` | 两个内核的权重带宽 | 只报数，无判据；与 `tools/hbm_bandwidth_probe.cu` 对照 |
| `pattern_attribution.cu` | 逐层加工作，看各自代价 | 只报数；用来判断还值不值得改 |
| `activation_quant_accuracy.cu` | int8 激活路径 vs bf16 路径 | 只报数；看 RMS 比与每行误差分布 |
| `nsys_decode_budget.py` | 解码阶段 GPU 时间预算 | 需要先跑 nsys，见下 |

### 为什么 `gemv_reference_check` 是门禁而不是可选

开发中出现过一次 tail 列映射写错：模型仍然生成**连贯的英文**，391 正控也过，只有 logits 是错的。
端到端冒烟完全没发现，是这个对拍抓出来的（`max_rel_err` 从 0.0033 跳到 92）。

覆盖的形状刻意包含边界：奇数行数（257、777，考验行数 guard）、非本模型宽度
（K=1280 → 10 组、K=3840 → 30 组）、本模型真实组数（5120→40、6144→48）。

### 带宽的两个测量陷阱

1. **载荷必须远大于 L2。** 本卡 L2 是 32 MB，而真实的 gate_up 张量只有 39 MB，迭代测量时会部分
   命中缓存而虚高。所以形状表里混了 78 / 156 / 312 MB。可达上限用仓里已有的
   `tools/hbm_bandwidth_probe.cu` 测（RTX 4060 Laptop 上 249.6 GB/s）。
2. **SM 频率随电源状态在 1.68–2.01 GHz 之间摆**，而内核是部分指令受限的，单次计时有约 ±20% 噪声。
   所以每项都跑 5 次取 best 与 median——与 `hbm_bandwidth_probe.cu` 同一口径。

### `pattern_attribution` 的结论（312 MB 冷载荷）

| 访存模式 | 带宽 |
|---|---|
| uint4 连续 | ~249 GB/s |
| 仅 code plane | ~249 GB/s |
| + activation | ~249 GB/s（激活是免费的，留在缓存里）|
| + high/scale 两条侧流 | ~196 GB/s（−22%）|
| + base-3 解码 | ~169 GB/s（−14%）|
| + dp4a 与 scale 累加 | ~120 GB/s（−29%，即真实内循环）|

**这是后续优化方向的关键依据**：剩余代价平摊在侧流、解码、累加三处，没有单一热点；
要再往上走得改权重布局（每行一个平面），而不是继续调度。参见开发文档 §9.1 五。

## nsys 时间预算

```bash
nsys profile --trace=cuda --cuda-graph-trace=node -o /tmp/trace \
  <build>/apps/ninfer <artifact> --max-new 32 --max-context 2048 \
  --kv-dtype rk4v4-e8 --no-thinking --greedy
nsys stats --report cuda_gpu_sum -o /tmp/stats /tmp/trace.nsys-rep
nsys stats --report cuda_gpu_trace --format csv -o /tmp/trace_gpu /tmp/trace.nsys-rep
python3 tools/verify/ternary_gemv/nsys_decode_budget.py \
  /tmp/stats_cuda_gpu_sum.csv /tmp/trace_gpu_cuda_gpu_trace.csv --tokens 32
```

`ms/token` 一列按解码步数摊薄。nsys 会把 100–400 µs 的内核时长抬高约 10–15%，
所以绝对值当上界看，真实速率以 `gemv_bandwidth` 为准。
