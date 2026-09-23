# 标准化跑分

`bench.sh` 是 `ninfer_bench` 的唯一入口。它存在的理由不是"省一次敲命令"，而是**可比性**：

- `ninfer_bench` 的默认语料是**相对当前工作目录**的 `bench/fixtures/bench_corpus.ids`。换一个目录跑，
  换的是语料，而输出里不留痕迹 —— 两次跑出来的数字不同，却看不出为什么。
- 重复次数、预热次数、prefill 分块都有各自的默认值，手敲一次很容易漏掉其中一两项。
- 结果默认打到标准输出，规模一多就没法归档，也没法和上一次对列。

本脚本把语料、重复、预热、分块钉死，把每条用例落成 tidy CSV，并把 GPU / 驱动 / CUDA / 引擎修订 /
制品摘要 / `NINFER_*` 开关一并写进 `manifest.txt`。**没有 `manifest.txt` 的跑分不作为证据。**

## 用法

```bash
just bench /data/Ternary-Bonsai-2-27B-ninfer/Ternary-Bonsai-2-27B-PQ2_0.ninfer
just bench <artifact> kv          # 换一组 suite
just bench <artifact> all         # 全部 suite
```

等价于直接调用（`just` 只是把路径钉好）：

```bash
NINFER_ROOT=/path/to/ninfer-4090 tools/bench/bench.sh <artifact.ninfer> [suite ...]
```

## suite

| suite | 内容 | 次数 |
|---|---|---|
| `standard`（默认）| `pp512` / `pp2048` / `tg128` / `pp512+tg128` | 1 次装载 |
| `prefill` | `pp512` / `pp2048` / `pp8192` | 1 次装载 |
| `decode` | `tg128` / `tg512` | 1 次装载 |
| `kv` | `bf16` / `int8` / `rk8v4` / `rk4v4` / `rk4v4-e8` / `rk2v4-e8`，各一条 `pp512+tg128` | 6 次装载 |
| `mtp` | 投机窗口 0 / 4 / 8 | 3 次装载 |
| `graph` | CUDA Graph 开 / 关 | 2 次装载 |
| `all` | 以上全部 | 14 次装载 |

同参数的用例合并进一次 `ninfer_bench` 调用：一个 20 GB 制品的装载时间与跑分本身同量级，
拆开跑等于把大部分时间花在重复装载上。

`mtp` 的 draft 8 显式带 `--no-cuda-graph`。这是**规避一个上游缺陷**，不是本移植的选择：草稿窗口
≥ 8 时 CUDA Graph 的 `cudaErrorGraphExecUpdateFailure` 会在官方非三元制品上同样复现。带图跑那条
用例测到的是缺陷，不是投机。

## 输出

```text
./out/bench-<制品名>-<UTC 时间戳>/
  manifest.txt             环境与输入摘要（制品/GHz/驱动/引擎修订/开关）
  results.csv              全部用例的 tidy CSV，首两列是 case,suite
  <suite>__<case>.csv      单次 ninfer_bench 的原始 CSV
```

`results.csv` 的首两列是脚本加的，用来在合并后的表里定位来源；其余列就是 `ninfer_bench -o csv` 的
原始列（`label,kind,n_prompt,...,decode_output_tok_s_mean,...`）。跨制品对列时，先确认两边的
`manifest.txt` 在 `prefill_chunk` / `kv_dtype` / `mtp_draft_tokens` / `weights_id` 上一致。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `NINFER_BENCH` | 在若干常见构建目录里找 | `ninfer_bench` 可执行文件 |
| `NINFER_ROOT` | — | 定位默认语料与记录引擎修订 |
| `NINFER_BUILD_ROOT` | — | 参与定位 `ninfer_bench` |
| `NINFER_BENCH_OUT` | `./out/bench-<制品名>-<时间戳>` | 输出目录 |
| `NINFER_BENCH_CORPUS` | `<NINFER_ROOT>/bench/fixtures/bench_corpus.ids` | 语料 |
| `NINFER_BENCH_REPS` | `5` | 重复次数 |
| `NINFER_BENCH_WARMUP` | `1` | 预热次数 |
| `NINFER_BENCH_CHUNK` | `1024` | prefill 分块（128 的倍数）|
| `NINFER_BENCH_DEVICE` | 不传 | CUDA 设备序号 |
| `NINFER_BENCH_EXTRA` | — | 追加到每条用例的参数 |
| `NINFER_BENCH_NO_HASH` | `0` | 置 1 跳过制品摘要（省一次全文件读取）|

## 本机实测（RTX 4090 / sm_89 / CUDA 13.3）

上表不是示例，是这套脚本在 `Ternary-Bonsai-2-27B-PQ2_0.ninfer` 上跑出来的（`-r 5 --warmup 1
--prefill-chunk 1024`，`manifest.txt` 里能查到制品摘要与引擎修订）：

| suite | case | prefill t/s | decode t/s | 备注 |
|---|---|---|---|---|
| standard | pp512 | 267.2 ± 18.4 | — | |
| standard | pp2048 | 303.4 ± 5.6 | — | |
| standard | tg128 | — | 50.8 ± 4.3 | |
| standard | pp512+tg128 | 397.9 | 48.1 | 同一次生成里的两段 |
| kv | bf16 / int8 / rk8v4 / rk4v4 / rk4v4-e8 / rk2v4-e8 | 427 / 327 / 332 / 383 / 263 / 290 | 41.6 / 43.4 / 45.7 / 44.2 / 41.5 / 44.8 | `-pg 512,128` |
| mtp | off / draft4 / draft8(无图) | 450 / 291 / 258 | 46.3 / 47.4 / 25.6 | 接受率 0% / 32.6% / 16.6% |
| graph | 开 / 关 | 396 / 267 | 49.4 / 24.2 | 图对 decode 的收益在这里最直白 |

两点值得留意，都是脚本帮你看出来的，不是它替你下的结论：

- **`pp512` 与 `pp512+tg128` 的 prefill 数不一样**（267 对 398）。前者只请求 1 个 token、后续
  decode 不参与计时，后者把两段都算上；跨表比较时别把它们当成同一个量。
- **`ninfer_bench` 口径的 MTP 接受率是 32.6%**，而 CLI 贪心口径是 74.4%。基准自己采样，且草稿
  开销算进墙钟，所以 draft4 的 decode 只从 46.3 抬到 47.4 t/s。两个数都对，量的不是同一件事。

## 已知的口径差异

`ninfer_bench` 是"产品路线"计时（`Engine::generate` 的公开路径），与 CLI 的贪心接受率口径不同。
实测中 MTP 在 `pp512/tg128` 上没有变快（39.0 对 44.3 tok/s），而 CLI 口径的接受率是 74–77%。
两者不矛盾 —— 前者把草稿开销算进墙钟，后者只统计被接受的 token —— 但**不要拿其中一个去否定另一个**。
要下"投机划不划算"的结论，需要按草稿窗口扫一遍 `pp/tg`，那是本脚本 `mtp` suite 能提供的输入，
不是它的结论。
