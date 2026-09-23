# 移植报告：NInfer 三元能力 → ninfer-4090

本文件记录这次适配的**判定依据与实测证据**。改动清单与重建步骤见
[../patches/README-改动说明.md](../patches/README-改动说明.md)。

---

## 1. 问题的形状

`ninfer-ada-ternary` 发布的是"工具与方法"：`patches/changed-files/` 是叠在
`Ambolio/ninfer-4090-windows @ 6eb70a07`（v1.0.8-windows 线）上的整文件快照。

目标是 `UDPSendToFailed/ninfer-4090 @ 5c60b7c9`（v1.2.0 线）—— 它能跑 sm_86 与 sm_89、支持 Linux，
但**不是同一条线**。两棵树在受影响文件上的差异（`diff` 行数）：

| 文件 | 三元增量 | v1.2.0 与 v1.0.8 的差异 |
|---|---|---|
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | 10 | **12110** |
| `src/targets/qwen3_6/impl/runtime/dflash_impl.h` | 6 | 423 |
| `src/ops/wrapper/gdn_input_proj.cpp` | 191 | 390 |
| `src/targets/qwen3_6_27b/impl/load/bindings.cpp` | 196 | 403 |
| `src/CMakeLists.txt` | 15 | 280 |

结论：**整文件覆盖不可行**。必须先把"真正的三元增量"从 v1.0.8 基线里分离出来。

## 2. 分离增量的方法

```
git clone --branch v1.0.8-windows --depth 1 <Ambolio/ninfer-4090-windows>  # 拿到 6eb70a07
diff -u <v1.0.8 原文件> <ada-ternary changed-files 同路径文件>              # 这才是三元补丁
```

这样得到 **约 1100 行真实改动 + 12 个新增文件**（相比"38 个文件全量覆盖"小了两个数量级），然后逐处
重新落到 v1.2.0 的对应代码上。每处的落点都以 v1.2.0 自己的上下文为准：例如 `variant.cpp` 里
权重档案枚举叫 `GroupwiseInt / GroupwiseIntW8Endpoints`（Ada 线叫 `Qwen36GroupwiseInt /
`Qwen38GroupwiseInt`）、`bind_weight()` 没有 `placement` 形参、`row_view()` 的每组字节数需要按
格式推导。

## 3. 判定依据：哪些地方"不能照抄"

判定规则只有一条：**目标线上是否存在这条路。** 不存在就不移植，而不是硬塞。

| 原补丁的做法 | v1.2.0 的现实 | 决定 |
|---|---|---|
| 在 `linear.cpp` / `embedding.cpp` / `linear_swiglu.cpp` 里为 `NVFP4`、`FP8_E4M3FN_ROW_BF16S` 加分支 | `QType` 只到 `I32_CTRL=6`，`src/ops/linear/` 下无 `nvfp4/`、`fp8/` | 丢掉这些分支 |
| `Weight` 带 `weight_scale_divisor` / `input_scale_divisor` | 本线 `Weight` 无这两个字段（NVFP4 专用）| 不引入 |
| 改 `program_impl.h` 给 `target_logprobs` 路径传工作区 | 本线**没有** `target_logprobs` / `score_hidden` | 该文件零改动 |
| CMake 放宽 `ninfer_media_acquire` 的条件 | 本线 `NINFER_BUILD_MEDIA_ACQUIRE` ⊇ `NINFER_BUILD_PROMPT_INPUT`，无悬空链接 | 不移植（与三元无关）|
| 旋转内核 `#include "ops/kv_cache/hadamard_d256.cuh"` | 本线删掉了 `ops/kv_cache/` | 补回上游同路径文件，保持三元文件逐字节一致 |
| `pack.py` 调 `row_split_geometry("PTQ1_0_G128", …)` | 本线 `QuantFormat` 按位宽反推每组字节，表达不了 24+2 / 32+0 | 新增 `TernaryFormat` 并注册 |

`QType` 的**取值**刻意保留 Ada 线的 9 / 10（而不是在本线补成 7 / 8）：两条线的日志与诊断要能对照着读，
且 7 / 8 在 Ada 线已被占用，本线将来并入 NVFP4 / FP8 时补空位会与本处静默冲突。

## 4. 实测证据

### 4.1 编译

- **C++**：对打过补丁的树逐翻译单元做 `g++ -std=c++20 -DNINFER_SM89=1 -fsyntax-only`，覆盖 `src/` 与 `apps/` 下全部 `.cpp`。
  失败 6 项，补齐 CMake 等效 include 路径后复测，只有 2 项是真缺系统包（`src/media/decode/decode.cpp` 缺 `libavcodec/avcodec.h`，
  `src/product/media_acquire/acquire.cpp` 缺 `curl/curl.h`）；`apps/serve/main.cpp` 与 `src/serve/responses_http.cpp` 用内置
  `third_party/cpp-httplib` 即可通过（原失败是探针漏传 `-I`），`src/serve/http_server.cpp` 缺的是 CMake 生成的 `ui.h`，
  `frontend.cpp` 缺的 `xgrammar` 由 configure 阶段 FetchContent 拉取。均与本补丁无关，详见 `docs/依赖安装-RockyLinux10.md`。
- **CUDA**：`nvcc 13.3 / sm_89` 编译 3 个三元翻译单元，全部通过；唯一告警是上游既有的
  `launch_pq2_gemv declared but never referenced`（本包注释里已说明该函数为何暂时不接线）。
- **完整 CMake 构建**：sm_89 与 sm_86 两条都从零 configure、全量编译并链接通过（`exit 0`），
  带 `-DBUILD_TESTING=ON` 的第三条也完成。细节与 cubin 证据见 §4.5。

### 4.2 旋转内核：真机 + 独立 oracle

```bash
NINFER_ROOT=<打过补丁的 ninfer 树> tools/verify/run_rotation_oracle.sh
```

harness 直接 `#include` 引擎同一份 `ternary_rotation_kernels.cuh`，所以编的是**真代码**；oracle 用
显式构造的 1024×1024 归一化 Sylvester-Hadamard 矩阵（已验证对称且正交）逐例比对。

| 用例 | k | tokens | 置换 | 方向 | 结果 |
|---|---|---|---|---|---|
| `plain_t1` | 5120 | 1 | 无 | 正向 | PASS rel_l2 2.2e-3 |
| `plain_t3` | 5120 | 3 | 无 | 正向 | PASS rel_l2 2.2e-3 |
| `perm_t1` | 6144 | 1 | (128,16,3) | 正向 | PASS rel_l2 2.2e-3 |
| `perm_t2` | 6144 | 2 | (128,16,3) | 正向 | PASS rel_l2 2.2e-3 |
| `wide_t1` | 17408 | 1 | 无 | 正向 | PASS rel_l2 2.2e-3 |
| `inv_t2` | 5120 | 2 | 无 | 逆向（词嵌入）| PASS rel_l2 1.5e-3 |

每个用例同时跑 3~5 个负控（符号与旋转次序对调、未归一化、符号行错位、置换方向反了、缺 P），
全部与原值分离（rel_l2 ≥ 0.97）—— 说明这个比对"有牙齿"。

### 4.3 这一节暴露出来的上游缺陷

第一次跑是 **3/6**：T=1 的三个全绿，**T>1 的三个全红**。原因不在内核，而在验证脚本：

- `oracle_rot.py` 用 `x.reshape(k, tokens)`（C 序）读缓冲，而引擎的 `[k, tokens]` 激活是
  **token 主序**（`ne[0]` 连续，元素 `(column, token)` 在 `token*k + column`）。T=1 时两种读法
  **完全重合**，所以只有 T>1 才会暴露。改为 `order="F"` 后 6/6。
- `check_embedding.py` 同一问题，且注释把布局写反了。
- `harness/gemm_test.cu` 少传后期新增的 `out_row_stride`，**根本编译不过**。
- `harness/rot_test.cu` 里 `return prop.name;` 返回局部变量地址。

这正好印证了原包 `docs/04` 里的那条告诫：*"任何 GEMM/变换类改动必须有 T>1 的用例，并在引擎侧验证；
T=1 时 token 主序与行主序完全重合，所以 T=1 全绿 ≠ 正确。"* —— 只是当时的独立 harness 自己没做到。
本包把这四处修掉后，独立 harness 第一次真正具备 T>1 的判别力。

### 4.4 制品几何

`tools/artifact` 与引擎侧 `storage_layouts.cpp::quant_geometry()` 的一致性由
`python -m ninfer_ternary check` 与 `tests/test_checks.py` 双重把守：`[248320, 5120]` 下
PTQ1_0_G128 = 278,118,400 B、PQ2_0_G128 = 337,715,200 B，与引擎注释里写死的数字逐字节一致。

### 4.5 架构支持：三元移植没有收窄任何东西

**先纠正一个前提：`ninfer-4090` v1.2.0 并不支持 `sm_86`~`sm_120`。** 它只支持 `sm_86` 与 `sm_89`，
而且是一道硬闸门（`CMakeLists.txt`，在任何 `option` 之前）：

```cmake
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89)$")
  message(FATAL_ERROR "NInfer supports CMAKE_CUDA_ARCHITECTURES=86 or 89; got ...")
endif()
```

`sm_86` 来自 `Don-Chad/ninfer-3090` 血缘（3090），`sm_89` 是本线主目标（4090）。传 `120` 会在
configure 阶段被直接拒绝 —— 这不是本补丁引入的，本补丁一行都没碰架构判定。

**NVFP4 / FP8 也不是本补丁丢弃的。** 它们是 v1.0.8-windows（Ada / Windows）那条线的格式：
本线的 `QType` 到 `I32_CTRL = 6` 为止、`src/ops/linear/` 下只有 `bf16 / q4 / q5 / q6 / w8`；
而来源基座 `ninfer-4090-windows @ v1.0.8` 里才是 `NVFP4 = 7`、`FP8_E4M3FN_ROW_BF16S = 8`，
外加 `src/ops/linear/{nvfp4,fp8}/`。这件事写在 v1.2.0 自己的发布说明里：

> **Blackwell SM120 and NVFP4 Purge** —— *Surgically deleted 45+ SM120 NVFP4 kernel files, TMA
> loaders, container parser descriptors, test fixtures, model cards, and conversion scripts,
> keeping the codebase strictly targeted to Ada Lovelace (`sm_89`).*
> —— `RELEASE_NOTES_1.2.0.md`

也就是说：NVFP4 / FP8 是**上游在 v1.2.0 里主动铲掉的**，比本补丁早存在一个版本。本补丁从 Ada 线
搬来的三元增量里确实夹带了一些 `case QType::NVFP4:` / `FP8_E4M3FN_ROW_BF16S` 分支，但那些分支引用
的是 Ada 线的枚举值 —— 在本线这两个符号**根本不存在**，照抄会直接编译不过。去掉它们是编译前提，
不是功能取舍；把它们"加回来"等于把整条 Ada 线的 NVFP4/FP8 内核族反向移植过来，那是另一个工程。

#### 条件编译本来就在，而且两个架构都真编过了

本仓没有另造开关：架构选择走的是上游既有的两处机制 —— `-DCMAKE_CUDA_ARCHITECTURES=86|89` 与由它
派生的 `NINFER_SM86` / `NINFER_SM89` 宏；`tools/verify/build.sh` 只是把它包成 `NINFER_ARCH=86|89`，
并顺手修掉了它自己 `-- <额外参数>` 未被消费的缺陷。

三元内核**没有引入任何架构专属代码**：三元目录下 grep 无 `__CUDA_ARCH__`、无 `NINFER_SM86` /
`NINFER_SM89`（全仓唯一用到这两个宏的是上游既有的 `w8_rowsplit_gemm_splitk.cu`）。
两个架构各自完整构建的结果：

| 构建 | 命令 | 结果 |
|---|---|---|
| sm_89（4090）| `NINFER_ARCH=89 tools/verify/build.sh clean` | **exit 0**；`apps/ninfer` 234 MB、`apps/ninfer-serve` 246 MB |
| sm_86（3090）| `NINFER_ARCH=86 tools/verify/build.sh clean` | **exit 0**；`apps/ninfer` 242 MB |
| sm_89 + 测试 | `NINFER_ARCH=89 … build.sh clean -- -DBUILD_TESTING=ON` | 见 §5 关于上游测试缺陷的说明 |

两个架构里三元内核都是**原生 cubin**（不是 PTX 兜底，也不会在 3090 上退回解释执行）：

```text
ternary_rotation.cu.o      -> ternary_rotation.sm_89.cubin      / ternary_rotation.sm_86.cubin
ternary_rowsplit_gemm.cu.o -> ternary_rowsplit_gemm.sm_89.cubin / ternary_rowsplit_gemm.sm_86.cubin
```

#### 想把 sm_120（5090）加回来要做什么

那是**反向移植**，不是改一个数字：需要把 v1.2.0 铲掉的 45+ 个 SM120/NVFP4 内核、TMA loader、
容器解析描述符从 `ninfer-4090-windows` v1.0.8 线搬回来，让 `QType` 扩到 7 / 8，并把 MMA 调度从
Ada 适配到 Blackwell；然后才谈得上摘掉上面那道 `FATAL_ERROR`。上游把它写成硬失败是有意的：
没有验证过的架构不该静默走进一套只做过 Ada 调优的调度。本补丁把三元取值放在 9 / 10、避开 7 / 8，
正是为了给这件事留出空位而不产生静默冲突。

### 4.6 上游漂移：官方模板制品已经走到容器 v3

在准备端到端验证时发现，`neroued/Qwen3.8-27B-NInfer` 的 `main` 分支上的 `qwen3_8_27b.ninfer`
**不是引擎能读的容器版本**。实测前 8 字节：

| 修订 | 首 8 字节 | 容器 | 目录偏移 | `json_bytes` | 体积 |
|---|---|---|---|---|---|
| `1cbd84e7`（`main`）| `NINFER\x00\x03` | **v3** | 32 | 372,704 | 20,437,521,664 B |
| `dc370fb6295a` | `NINFER\x00\x02` | **v2** | 16 | 185,105 | 20,437,336,576 B |

v3 在前缀之后多了 16 字节摘要，JSON 目录从偏移 32 开始，且顶层键变成
`components / objects / bindings / uses / metadata / provenance / files`（**没有** `identity`）。
而 ninfer-4090 v1.2.0 两边都只认 v1 / v2：

- C++：`src/artifact/reader.cpp` 的 `kMagic` / `kV1Magic`，否则抛 `artifact magic is not NInfer v1 or v2`；
- Python：`tools/artifact/container.py` 的 `MAGIC = b"NINFER\x00\x02"`。

时间线也对得上：HF 提交 `51630a0c` 就是「Publish v3 artifact and updated model card」，在 v1.2.0 之后。
上一个修订 `dc370fb6295a`「Update artifact with DFlash2 companion weights」是 v2，并且正好带 66 个
`dflash2/*` 张量 —— 即 v1.2.0 新增的 validate-only stub 要消费的那批。**做三元模板应该钉这个修订。**

`pack.py` 现在会先读容器前缀并把版本号直接报出来，而不是在偏移 16 上解一个不是 JSON 的东西、
抛一个与真实原因无关的 `JSONDecodeError`。

### 4.7 端到端：两个格式都真的答对了

用真实权重（`/data/Ternary-Bonsai-2-27B-gguf`）转出两个制品，在本机 RTX 4090 上跑通：

| 制品 | 体积 | 对象 | 引擎装载 | `17 * 23` 贪心输出 | decode |
|---|---|---|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0.ninfer` | 10,533,732,876 B = 9.810 GiB | 1192 | 772 张量 / 6 资源，权重 6.70 GiB | **391** | 52.3 tok/s |
| `Ternary-Bonsai-2-27B-PTQ1_0.ninfer` | 9,274,212,876 B = 8.637 GiB | 1192 | 772 张量 / 6 资源，权重 5.52 GiB | **391** | 15.8 tok/s |

两个制品各 1192 个对象 = 模板 1190 + 新增的 `text/hadamard_signs` / `text/hadamard_widths`；
借用模板 3.114 GiB（`frontend` 6、`text` 2、`mtp` 12、`vision` 333、`dflash2` 66）。

#### 两个开关的双向对照（这才是关键）

| 运行 | PQ2_0 | PTQ1_0 | 判读 |
|---|---|---|---|
| `NINFER_TERNARY_MMA=1` | `391` | `391` | 张量核路径 |
| `NINFER_TERNARY_MMA=0` | `391` | `391` | SIMT 路径；**与上面逐字节一致** |
| `NINFER_TERNARY_HADAMARD=0` | 乱码 | 乱码 | **负控分离**：关掉折叠基旋转立刻崩坏 |

这三行合起来说明三件事：

1. 两条内核路径（MMA / SIMT）在本模型上给出**完全相同**的贪心序列 —— 此前"MMA 只做过编译验证"
   的空白在这里补上；
2. 关掉旋转就彻底崩坏，证明折叠基旋转是**载荷路径**而不是可选项，也证明 `391` 不是碰巧；
3. 两种三元解码（PQ2_0 的 2-bit 码、PTQ1_0 的 base-3 三元 + 高位平面）都正确 —— 它们走的是不同的解码原子。

PTQ1_0 比 PQ2_0 慢 3.3 倍（15.8 vs 52.3 tok/s）符合预期：PTQ1_0 每组 26 字节装 128 个权重且要做
base-3 的除法与取模，PQ2_0 每组 34 字节、纯移位取码。

复现命令（`--no-thinking --greedy` 让两条路径逐字节可比）：

```bash
NINFER_TERNARY_MMA=1 /data/ninfer-build/apps/ninfer <artifact.ninfer> \
  --prompt "What is 17 * 23? Answer with just the number." \
  --max-new 16 --max-context 512 --no-thinking --greedy --seed 1234
```

### 4.8 一致性矩阵、MTP 与 KV 量化

#### 内核路径 × prefill 分块 × 制品格式

同一 prompt（bat-and-ball）、`--no-thinking --greedy --seed 1234 --max-new 192 --max-context 2048`，
比较 CLI 的 `--print-token-ids` 给出的 token 序列：

| 用例 | PQ2_0 | PTQ1_0 |
|---|---|---|
| `NINFER_TERNARY_MMA=1` + `--prefill-chunk 128` | `9ab7f3dfe667` | `9ab7f3dfe667` |
| `NINFER_TERNARY_MMA=1` + `--prefill-chunk 1024` | `9ab7f3dfe667` | `9ab7f3dfe667` |
| `NINFER_TERNARY_MMA=0` + `--prefill-chunk 128` | `9ab7f3dfe667` | `9ab7f3dfe667` |
| `NINFER_TERNARY_MMA=0` + `--prefill-chunk 1024` | `9ab7f3dfe667` | `9ab7f3dfe667` |
| 负控 `NINFER_TERNARY_HADAMARD=0` | `2c037e1df050` | `0e7f23175819` |

10 次正控全部落在同一个 158 token 的摘要上；两个负控都分离。同一配置连跑 3 次也是同一摘要，
所以这个比对本身是可复现的，不是抖动。一键复跑：

```bash
NINFER_CLI=/data/ninfer-build/apps/ninfer tools/verify/e2e_ternary.sh <artifact.ninfer>
```

#### MTP 投机

| 配置 | token 序列 | 接受率 | 接受长度 |
|---|---|---|---|
| 无投机 | `9ab7f3dfe667` | – | – |
| `--spec mtp --draft-tokens 4` | `9ab7f3dfe667` | 74.38% | 3.98 tok/轮 |
| `--spec mtp --draft-tokens 4 --lm-head-draft` | `9ab7f3dfe667` | 76.92% | 4.08 tok/轮 |
| `--spec mtp --draft-tokens 8 --lm-head-draft --no-cuda-graph` | `9ab7f3dfe667` | 58.48% | 5.68 tok/轮 |

投机不改变贪心输出 —— 这是投机解码必须成立的性质，这里成立。

#### 上游缺陷：MTP 与 CUDA Graph 在较大草稿窗口下更新失败

`--spec mtp --draft-tokens 8`（及更大）在 `--max-new 192 --max-context 2048` 下会中止：

```text
error: CUDA Graph executable update failed: cudaErrorGraphExecUpdateFailure (update result 2)
```

加上 `--no-cuda-graph` 即恢复，且输出与基准逐字节一致。**这不是本补丁引起的**：同样的边界在
官方非三元制品 `qwen3_8_27b.v2.ninfer` 上逐条重现（≤7 通过、≥8 失败，两个制品的边界与报错完全一致）。
窗口 8 以上的失败还与配置有关 —— `Say hi.` + `--max-new 64` 下 8..15 全部通过 —— 所以这是 CUDA Graph
拓扑与"草稿窗口 × 上下文/生成长度"组合的问题，不是单纯的阈值。

#### KV 量化（PQ2_0，pp512 / tg128，3 次重复 + 1 次预热）

| `--kv-dtype` | pp512 t/s | tg128 t/s |
|---|---|---|
| `bf16`（默认）| 196.9 ± 62.0 | 67.4 ± 7.8 |
| `int8` | 274.8 ± 27.6 | 61.2 ± 7.6 |
| `rk8v4` | 261.1 ± 5.0 | 48.1 ± 4.3 |
| `rk4v4` | 258.4 ± 0.6 | 50.5 ± 8.4 |
| `rk4v4-e8` | 256.4 ± 15.9 | 44.3 ± 1.6 |
| `rk2v4-e8` | 265.2 ± 5.4 | 48.9 ± 5.3 |
| `rk4v4-e8` + MTP4 + 优化草稿头 | 272.5 ± 23.7 | 39.0 ± 3.0 |

六种 KV 存储都能在折叠三元制品上装载并跑完 pp512/tg128。注意 `ninfer_bench` 的 `--kv-dtype` 取值集合
是 `bf16 | int8 | rk8v4 | rk4v4 | rk4v4-e8 | rk2v4-e8`，与 CLI 的 `--kv-dtype` 命名不同（`e8` 不是合法值）。

#### 长上下文一致性

把 prompt 拉到 2685 与 11043 token，扫 prefill 分块与内核路径：

| prompt token | 分块 | MMA=1 摘要 | MMA=0 摘要 | prefill（MMA / SIMT）|
|---|---|---|---|---|
| 2685 | 512 | `e648b9c1c25b` | `e648b9c1c25b` | 262.2 / 48.8 t/s |
| 2685 | 1024 | `e648b9c1c25b` | `e648b9c1c25b` | 274.8 / 60.1 t/s |
| 2685 | 2048 | `e648b9c1c25b` | `e648b9c1c25b` | 287.3 / 66.1 t/s |
| 11043 | 1024 | `41907670335c` | `41907670335c` | 282.1 / 62.8 t/s |
| 11043 | 2048 | `41907670335c` | `41907670335c` | 282.6 / 71.6 t/s |

两条 prompt 都要求从长上下文里取回一个事实，答案分别正确：2685 token 那条答
"The fox jumped over the lazy dog."，11043 token 那条答 "The dog is lazy."。
也就是说 1 到 11 个 prefill 分块之间的切换、以及张量核与 SIMT 两条路径之间，都没有可观测差异。

顺带得到一个量化结论：**三元 MMA 路径的 prefill 是 SIMT 的约 4.2~4.4 倍**（282 vs 63~72 t/s）；
decode 端反而只有约 +8%（52.3 vs 48.0 t/s），因为 decode 是 T=1 的带宽受限 GEMV。

越界时行为也是明确的：prompt 超过 `--max-context` 时引擎报
`error: prepared prompt exceeds Engine context capacity` 并以非零码退出，而不是静默截断。

#### 并发服务

`ninfer-serve --max-concurrency 4`，8 个请求同时到达（4 个槽位，实际 decode 批 3.76）：

| 项 | 结果 |
|---|---|
| HTTP 状态 | 8/8 **200** |
| 生成内容 | 8 条**逐字节一致**（摘要 `0fd9b946f31b`）|
| 单请求时延 | TTFT 381~2669 ms，wall 2.34~3.84 s（48 token）|
| decode 吞吐 | 每请求约 41 t/s |

并发下贪心输出仍然一致，说明折叠三元路径在批量 decode 槽位里也是确定的。

### 4.9 本补丁自己引入的回归：从 82/84 到 84/84

`ctest` 曾有两项红色，**根因在本补丁**而不是上游。写在这里，因为"哪些红是上游的、哪些是自己的"
决定了这个补丁能不能被接受。

#### 症状

`ninfer_gdn_input_proj_conv_snapshot_test` / `ninfer_gdn_input_proj_conv_record_test` 报
`workspace query/execution high-water mismatch`。

折叠三元的父权重与 Q4/Q5 共用同一套**行几何**（q/k 2048、v 6144），而两条容量查询
`gdn_input_proj_conv_{snapshot,record}_workspace_capacity_bytes` 的签名里只有形状。最初的移植选择
"让两条查询按三元的最坏情况无条件预留"，代价是 groupwise-int 制品也被多留一块 `[5120, T]`
激活缓冲 —— 那两个测试断言的正是 `peak_used() == 查询值`，多留即失败。

#### 修法

1. 新增 `WeightsProfile::FoldedTernary`（`package.h`），`resolve_weights()` 依制品身份
   `qwen3.8-27b/folded-ternary` 解析它。
2. 两条 ops 容量查询**恢复上游实现**，三元另开两条专属查询
   `gdn_input_proj_conv_{snapshot,record}_folded_workspace_capacity_bytes`。
3. `Variant` 的每个携带权重档案的容量查询按档分派：groupwise 档回到上游精确值，折叠三元档保留
   旋转 scratch。`gdn_norm_control_projection_workspace_capacity_bytes` 是唯一例外 —— 它的签名被
   运行时模板与 35B 目标共用，而那一档没有折叠三元，所以保留无条件预留。

#### 一个被"只改规划期"这个前提掩盖的陷阱

第一版修法只动了**规划期**的容量查询，`ctest` 立刻回到 84/84，但 `--spec mtp` 全线崩在
`std::bad_alloc`，而 `ctest` 覆盖不到它。原因不在规划期：

`Variant::gdn_input_projection_record` 会为 record 路径**开一块借来的叶子竞技场**：

```cpp
const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
WorkspaceArena leaf_workspace(storage);
```

上游的 `gdn_input_proj_conv_record_workspace_capacity_bytes` 对融合的 Q4/Q5 返回 0，所以
`gdn_record_workspace_bytes` 得到 `max(1, 0) = 1` —— 叶子竞技场只有 **1 字节**，而折叠三元的算子
要在里面分配 `[5120, 5]` 的旋转缓冲（51200 字节）。`DeviceArena::alloc_bytes` 在越界时抛的正是
`std::bad_alloc`（`src/core/arena.cu`），于是 MTP 的 verify 阶段在**图构建**时中止。

这条路径在**运行期**，拿不到权重档案，只能看权重自己声明的格式。所以修法是让
`gdn_record_workspace_bytes` 按 `weight.qtype` 判断，并把两块需求取 `max` —— 取 `max` 而不是相加：
规划期给这块叶子预留的就是两者中的较大者，多算一字节一样是超容量。

定位过程值得记一笔：把 `DeviceArena::alloc_bytes` 的越界分支临时改成打印
`bytes / aligned_offset / cap`，一眼就看到 **`cap=1`** —— 那正是 `kMinimumLeafWorkspaceBytes`。
一次插桩省掉了照着 84 个用例猜的时间。

#### 代价：制品必须重打

`pack.py` 从这一版起把产物的 `identity.weights_id` 写成 `folded-ternary`（模板仍必须是
`groupwise-int`）。**改动之前打好的三元制品不能再用**：它们声明 `groupwise-int`，会被解析成
非三元档，于是旋转缓冲**不被预留**。引擎对 `qwen3.8-27b` 只接受两种身份组合，其余在
`resolve_weights()` 直接抛错。

#### 复验

| 项 | 修复前 | 修复后 |
|---|---|---|
| `ctest` | 82/84（2 项失败）| **84/84**（5 项 `real` 按设计跳过）|
| 制品身份 | `qwen3.8-27b/groupwise-int` | `qwen3.8-27b/folded-ternary` |
| 制品尺寸 | PQ2_0 10,533,732,876 B；PTQ1_0 9,274,212,876 B | **字节数完全相同** |
| 端到端正控摘要 | `9ab7f3dfe667`（5 条全同）| **`9ab7f3dfe667`，逐位不变** |
| 端到端负控摘要 | PQ2_0 `2c037e1df050`；PTQ1_0 `0e7f23175819` | 同前，逐位不变 |
| MTP 投机（draft 4 + 优化草稿头）| 接受率 74.38%（PTQ1_0）| 76.92%（PQ2_0）/ 74.38%（PTQ1_0），输出与无投机逐字节一致 |
| 旋转 oracle | 6/6 | 6/6 |
| 非三元（官方制品）路径 | — | `qwen3_8_27b.v2.ninfer` 正常装载，`The capital of France is` → **Paris**；分块 128 / 1024 输出逐字节一致；**逐条**容量查询回到上游精确值（工作区**总容量**两档实测相同，见《权重档案与容量规划》§6.1）|

修复的判据不止 `ctest` 转绿：**两种三元格式的 5 条正控摘要 `9ab7f3dfe667` 与两个负控摘要，
在修复前后逐位相同**，制品的字节数也一模一样 —— 说明这次改动只动了"留给多少临时字节"，没有动
任何一个算出来的数。而 `ctest` 与端到端都没覆盖到的 `--spec mtp`，是这次修复真正的验收项。

> `ctest` 以 `-j 8` 并行跑时，`ninfer_state_store_test` 与 `ninfer_disk_state_cache_test` 会偶发失败
> （多轮里大部分轮次 84/84，失败的那一项每次还不一样）。两者都操作磁盘状态、没有资源锁，单独跑
> 必过，且不在本补丁的改动范围内 —— 这是上游的并行测试缺陷，与本移植无关。

以本节提到的权重档案为题的独立说明见 [《权重档案与容量规划》](权重档案与容量规划.md)：
逐条列出七条按档分派的容量查询与一条例外、三个方向的失效模式，并用配对跑分回答"档案到底改不改变
总容量"。

### 4.10 干净检出可复现：从 clone 走到 ctest

前面各节都是在"已经打完补丁的工作树"上测的。这一节回答另一个问题：**别人拿到本仓从头跑一遍，
能不能得到同一棵树、同一套结果？**

```bash
git clone /root/ninfer-4090 /root/ninfer-fresh      # HEAD 是未改动的 v1.2.0（5c60b7c9）
uv run python -m ninfer_ternary apply --repo /root/ninfer-fresh
diff -r --exclude=.git /root/ninfer-fresh /root/ninfer-4090
```

`apply` 写入 45 个文件，`diff -r` **无差异**，`check` 20/20 —— 工作树逐字节可重建。

第一次做这个核对时 `diff` 报出一处差异，那正是这一节的价值所在：`tests/targets/qwen3_6_27b/test_load_plan.cpp`
用了 `std::ranges::count_if` 却没包含 `<algorithm>`。gcc 14.3.1 + libstdc++ 15（Rocky Linux 10）
不再传递包含它，单翻译单元 `-fsyntax-only` 直接报 `'count_if' is not a member of 'std::ranges'`。
这一行当时只活在本机工作树里、**不在补丁清单中**，意味着任何人干净检出后跑 `just build-tests`
都会断在测试目标上 —— 而 `just ctest` 是本包验证流程的一环。现已把该文件作为第 45 项纳入清单。

| 步骤（全部在干净检出上）| 结果 |
|---|---|
| `python -m ninfer_ternary apply --repo <fresh>` | 写入 **45** 个文件；`diff -r` 与已应用检出**无差异** |
| `python -m ninfer_ternary check --repo <fresh>` | **20/20** |
| `build.sh incremental -- -DBUILD_TESTING=ON` | **726/726 目标，exit 0**（含此前编不过的 `ninfer_qwen3_6_27b_load_plan_test`）|
| `ctest` | **84/84 通过，0 失败**（5 项 `real` 按设计跳过）|

于是 `just from-scratch PQ2_0`（补丁 → 编译 → 测试 → 打包）是一条真能从零跑到尾的命令，而不是
"先手工把文件覆盖进去再说"。

### 4.11 装成工具：`uv tool install` 走完同一条链路

前面各节都在仓库里跑。这一节回答另一个问题：**不克隆本仓，能不能得到同一个引擎？**

跑法就是一条命令（`just tool-install` 是它的等价物）：

    time uv tool install --force .

本仓自带一个 PEP 517 构建后端（`build_backend.py`），把 §4.10 的链路搬进了 wheel 构建阶段：
按清单钉死的提交取上游（先试单提交浅取，失败退回完整克隆）→ 落地同一份 45 文件补丁 →
落地自检 → `cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89` 编译 `ninfer` 与 `ninfer-serve` →
把可执行文件、上游 `tools/artifact`、打包器与补丁打进 wheel → 删除整棵临时树。

| 项 | 实测 |
|---|---|
| 安装耗时 | **6 分 31 秒**（user 49 分 05 秒；本机 128 核，`-j 128`）|
| wheel | `ninfer_ternary-0.3.0-py3-none-linux_x86_64.whl`，**234,295,913 B = 223 MiB** |
| 装好的工具环境 | **517 MiB**（引擎本体）；带 `[convert]` 时 5.0 GiB（多一个 CUDA 版 torch）|
| 临时树 | 构建期只在 `$TMPDIR/ninfer-ternary/` 这一个根下开 `wheel-*` 与 `build-*`；编译子进程的 `TMPDIR` 指到 `build-*/tmp`，实测 nvcc 的 `tmpxft_*` 全落在那里。装完 `ls /tmp/ninfer-ternary* /tmp/tmpxft_*` 皆为空（空的根目录也会被删掉），`/tmp` 总量回到 10 MiB |
| 装好的 `ninfer` | `--prompt '17 * 23 =' --no-thinking` 输出 **391**，exit 0 |
| 装好的 `ninfer-convert` | 不设 `NINFER_ROOT`、只用随包的上游制品模块跑完 `check`：真实 20 GiB 模板 + 7 GiB GGUF，**1 分 55 秒**，`RESULT: OK` |

**为什么值得单开一节**：这是"这个包能不能独立存在"的判据。前面各节的结论都依赖
`/root/ninfer-4090` 这棵打过补丁的树；这条路径证明补丁、上游 Python 模块与编译产物可以被装进
一个 wheel，装完把源码树删掉也不影响使用。

两个诚实的缺口：

- **第一次安装卡在 WebUI 下载**。CMake 配置阶段要从 GitHub release 取 3 MiB 的服务端 UI 包，
  第一次尝试在 1 MiB 处停了 7 分钟（`file(DOWNLOAD ... TIMEOUT 60)` 对"缓慢但在推进"的传输
  不触发超时）；中止后重试，同一下载 1.2 秒完成。属于网络抖动，但说明安装流程暴露在 GitHub CDN
  上：`NINFER_TERNARY_ENABLE_UI=0` 可跳过它，代价是 `ninfer-serve` 用空 UI 桩。
- **`ninfer-convert` 需要 torch**。上游 `tools/artifact/layouts.py` 在**导入期**就用 torch
  （dtype 表与 Plane / Payload 类型别名），而本仓 `pack.py` 自己只把 torch 用在少数
  函数里。折中做法是放成额外项：`uv tool install ".[convert]"`，默认安装因此保持"只有引擎"；
  不带这个额外项时 `ninfer-convert` 会直接报缺 torch，而不是抛 traceback。

## 5. 未能验证的部分（诚实交代）

- **PPL / 困惑度没有测**，而且用本仓现有工具测不了：CLI 没有 logprob / score 选项，`eval/` 是
  EvalScope 驱动的任务评测（IFBench / AIME / GPQA），不产出困惑度；要测需要另写一个打分入口。
  原先的 PPL 6.445 来自 Ada / Windows 线，**不能直接外推到本线**。
- **上下文只压到 11k token**。§4.8 在 2685 与 11043 token 的 prefill 上逐字节一致，并发 4 槽也一致；
  但 README 里那套 400k 上下文的配置（`rk4v4-e8` 等量化 KV + 长程 prefill）没有复现，
  128k 量级的 prefill 与 `--max-concurrency > 4` 未覆盖。
- **~~引擎自带测试有 2 项失败~~ —— 已修复，现在 84/84 通过**（5 项 `real` 用例按设计跳过）。
  那 2 项失败确实由本补丁引起、不是上游：折叠三元与 Q4/Q5 共用行几何，而两条 ops 容量查询的签名里
  只有形状，于是最初选择"都按三元的最坏情况预留"，破坏了"查询值 == 执行高水位"这条不变量。
  修法是给折叠三元一档独立的 `WeightsProfile`，让容量查询按档分派；**代价是制品必须重打**
  （身份从 `groupwise-int` 改为 `folded-ternary`）。完整过程见 §4.9。
- **MTP 在基准口径下没有收益**：`ninfer_bench` 的 pp512/tg128 组合里 `rk4v4-e8` + MTP4 的 tg128 是
  39.0 t/s，反而低于不开投机的 44.3 t/s；这与 CLI 贪心下 74–77% 的接受率口径不同（基准自己采样）。
  MTP 在本机这条线上到底划不划算，需要单独一轮基准才说得清，本次没有下结论。
