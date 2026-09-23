# 三元 Bonsai → ninfer-4090 · 源码改动包

> **本包是把 `ninfer-ada-ternary` 的三元能力适配到 `ninfer-4090`（v1.2.0 线，sm_86 / sm_89，支持 Linux）的可移植快照。**
> 按目录名覆盖回 ninfer 源码树根目录 → **必须重编**（见 §D）→ 按 §E 做验证。
> 用 `python -m ninfer_ternary apply --repo <ninfer 树> --dry-run` 可以先看会写哪些文件。

| 项 | 值 |
|---|---|
| **目标树** | `UDPSendToFailed/ninfer-4090` @ `5c60b7c9b455231795c09da21a9fbb6aa53f08e5`（v1.2.0 线）|
| **改动来源** | `shensanshu/ninfer-ada-ternary` @ `ca845a4` |
| **来源基座** | `Ambolio/ninfer-4090-windows` @ `6eb70a07`（v1.0.8-windows，Ada / sm_89 / Windows）**——不是本目标树** |
| **文件数** | 45（新增 14，修改 31）|
| **产物** | `changed-files/`（整文件快照，覆盖即可）+ `0001-ternary-port-on-ninfer-4090.patch`（统一 diff，仅供审阅）+ `manifest.json`（逐文件摘要）|

---

## A. 为什么要"适配"而不是"覆盖"

`ninfer-ada-ternary` 的补丁是针对 **另一条线** 写的：

```
v1.0.8-windows（Ada / Windows 线）        v1.2.0（4090 / 3090 线，本目标）
  QType: … NVFP4=7, FP8=8                    QType: … 到 I32_CTRL=6 为止
  src/ops/linear/{q4,q5,q6,w8,bf16,nvfp4,fp8}  src/ops/linear/{q4,q5,q6,w8,bf16}
  src/ops/kv_cache/hadamard_d256.cuh          （无此文件）
  runtime 里有 target_logprobs / score_hidden  （无此路径）
  program_impl.h ≈ 12.4k 行                  program_impl.h = 2790 行
```

所以直接覆盖 38 个文件是不可能的。**本包的做法是：从 v1.0.8 基线取出"真正的三元增量"，逐处重新落到
v1.2.0 的对应位置上。** 增量本身很小 —— 除新增文件外，全部修改合计约 1100 行改动，其中相当一部分是注释。

---

## B. 改动清单（按层）

### B1. 新增：三元内核与旋转（14 个文件）

| 文件 | 作用 |
|---|---|
| `src/ops/linear/ternary/ternary_rowsplit_storage.cuh` | 两种格式的组几何 + 两个解码原子（PTQ1_0 base-3 移植、PQ2_0 2-bit）|
| `src/ops/linear/ternary/ternary_row_view.h` | 三元父权重的行切片（按 qtype 推几何，不是套用 Q4/Q5 的 32/0）|
| `src/ops/linear/ternary/ternary_rotation.{h,cpp,cu}` | 旋转的公开入口、开关、工作区尺寸、`folded_activation()` |
| `src/ops/linear/ternary/ternary_rotation_kernels.cuh` | 自包含 D1024 归一化 SWHT + 显式符号 + P 置换（正向 / 逆向就地），供独立 nvcc 单测复用 |
| `src/ops/linear/ternary/ternary_rowsplit_gemm.{cu,cuh}` | token 主序参考 GEMM（T=1 走 warp-per-row GEMV，T≥2 走张量核）|
| `src/ops/linear/ternary/ternary_rowsplit_gemv.cuh` | 解码用 warp-per-row GEMV + 小 token tile 变体 |
| `src/ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh` | T=2..8 的 PQ2_0 张量核路径（8-token tile，同时覆盖 verify 与 prefill）；`NINFER_TERNARY_MMA=0` 回退 |
| `src/ops/linear/ternary/ternary_dispatch.{h,cpp}` | 调度：先旋转进折叠基，再 GEMM |
| `src/ops/linear/ternary/ternary_launch.h` | `TernaryLaunch` 函数指针类型 |
| `src/ops/kv_cache/hadamard_d256.cuh` | **本线缺失**：上游同路径的 D256 构件（`hadamard_d32_columns_inplace`）。旋转内核复用它，改一行 include 也能挪位，但保持上游路径可让三元文件与本包之外的上游逐字节一致 |

### B2. 修改：格式注册与制品层

| 文件 | 改动 |
|---|---|
| `src/core/tensor.h` | 新增 `QType::PTQ1_0_G128 / PQ2_0_G128`；`Weight` 新增 `hadamard_signs / hadamard_n_blk / hadamard_perm_{hd,nk,rep}` |
| `src/artifact/reader.{h,cpp}` | 新增 `NumericFormat::PTQ1_0_G128 / PQ2_0_G128` 与名字解析 |
| `src/artifact/storage_layouts.cpp` | `quant_geometry()` = `{128,24,2}` / `{128,32,0}`；`format_name()` |
| `src/artifact/typed_binding.cpp` | 布局映射 + `qtype_for()` |
| `src/artifact/binder.{h,cpp}` | `Binder::find()`：**不消耗**地窥视对象描述符 |
| `src/CMakeLists.txt` | 加入 4 个三元源文件 |

### B3. 修改：算子接线

| 文件 | 改动 |
|---|---|
| `src/ops/linear/linear.cpp` | 三元 qtype 走 `ternary_dispatch`；工作区容量返回旋转 scratch |
| `src/ops/launcher/embed_gather.{h,cu}`、`src/ops/kernel/embed_gather.cuh` | 三元词表查表内核 + 两个启动器（复用同一套解码原子）|
| `src/ops/wrapper/attn_input_proj.cpp` | 折叠三元双父权重的四条投影共用一次旋转；新增带工作区的重载 |
| `src/ops/wrapper/gdn_input_proj.cpp` | 同上（qk / value / z 三路），并覆盖 conv-snapshot / conv-record / batch 三条路径。容量查询按权重档案分开：上游两条 `*_workspace_capacity_bytes` 保持原样，折叠三元另开两条专属查询（`*_folded_workspace_capacity_bytes`）|
| `src/ops/wrapper/linear_add.cpp` | 三元 = 共享 GEMM 到 scratch + 现成 `residual_add` |
| `src/ops/wrapper/linear_swiglu.cpp` | 三元 = 整块 gate_up GEMM + 现成 `silu_mul` |
| `src/ops/wrapper/embedding.cpp` | 三元词表查表后**就地**施加逆变换；`NINFER_TERNARY_DUMP_EMBED` / `NINFER_TERNARY_TRACE_EMBED` 诊断钩子 |

### B4. 修改：target / 运行时

| 文件 | 改动 |
|---|---|
| `src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h` | 新增 `WeightsProfile::FoldedTernary` 与制品身份常量 `folded_weights_id = "folded-ternary"` |
| `src/targets/qwen3_6_27b/impl/package.cpp` | `resolve_weights()` 把 `qwen3.8-27b/folded-ternary` 映射到该档 |
| `src/targets/qwen3_6_27b/impl/load/bindings.{h,cpp}` | 按**制品自己声明的格式**解析分组 row-split 权重；`text/hadamard_signs` + `text/hadamard_widths` 符号表；`/gdn/output` 的折叠置换。折叠三元档与 groupwise 档**共用同一张对象表**——绑定层从不看权重档案，档案只决定规划期的临时容量 |
| `src/targets/qwen3_6_27b/impl/variant.cpp` | 每个携带权重档案的容量查询**按档分派**：groupwise 保持上游精确值，折叠三元另计旋转 scratch；两处 split 投影改走带工作区的重载。record 路径的**叶子竞技场**容量由 `gdn_record_workspace_bytes()` 在**运行期**按 `weight.qtype` 算 —— 那条路径拿不到权重档案，规划期的分派管不到它，漏掉就会让叶子只剩 1 字节并在图构建时抛 `std::bad_alloc` |
| `src/targets/qwen3_6/impl/runtime/{text_context_impl.h,text_prefill_impl.h,dflash_impl.h}` | LM head 改用带工作区的 `linear()` |
| `tests/targets/qwen3_6_27b/test_load_plan.cpp` | **与三元无关的一行修复**：该文件用 `std::ranges::count_if` 却没包含 `<algorithm>`。gcc 14.3.1 + libstdc++ 15（Rocky Linux 10）不再传递包含它，`BUILD_TESTING=ON` 直接编不过（`'count_if' is not a member of 'std::ranges'`）。**不带上这一行，"干净检出 → 打补丁 → `just build-tests`"这条链路就断在测试目标上**，而 `just ctest` 是本包验证流程的一环 |

---

## C. 与 `ninfer-ada-ternary` 的**刻意差异**

下面每一条都是"不照抄"，理由随行。

1. **丢掉 NVFP4 / FP8 相关适配。** v1.2.0 线没有这两个格式（`QType` 到 `I32_CTRL=6`，`src/ops/linear/` 下无 `nvfp4/`、`fp8/`）。原补丁里所有 `case QType::NVFP4:` / `FP8_E4M3FN_ROW_BF16S` 分支、`weight_scale_divisor` 字段、`input_scale_divisor_bits` 都被去掉。
2. **`QType` 取值保留 9 / 10，而不是补成 7 / 8。** 两条线共用同一套日志与诊断输出，取值一致才能对着 log 说话；而且 7 / 8 在 Ada 线已被 NVFP4 / FP8 占用，本线将来若并入那两个格式，补空位会与本处静默冲突。`core/tensor.h` 里有注释说明。
3. **`src/targets/qwen3_6/impl/runtime/program_impl.h` 一个字都没动。** 原补丁改它是为了给 `target_logprobs` / `score_hidden` 那条打分路径的 output head 传旋转 scratch —— 本线**没有**这条路径（两个 27B 版本分别 2790 行 / 无 `target_logprobs`）。硬套会引入一条死代码。
4. **没有移植 `ninfer_media_acquire` 的 CMake 条件放宽。** 那条改动是 Windows fork 的构建修复，与三元无关；本线的 `NINFER_BUILD_MEDIA_ACQUIRE` 在 `NINFER_BUILD_APPS OR BUILD_TESTING` 时为 ON，是 `NINFER_BUILD_PROMPT_INPUT` 与 `ninfer_serve` 的超集，本来就不会出现悬空链接。
5. **新增 `src/ops/kv_cache/hadamard_d256.cuh`。** 旋转内核 `#include "ops/kv_cache/hadamard_d256.cuh"`，而本线删掉了 `ops/kv_cache/`。补回上游同路径的文件，三元那 13 个文件就能与本包之外的上游逐字节一致，省掉一处 include 改写。
6. **`tools/artifact` 补了三元格式注册。** 原包只发了 `src/` 改动，`pack.py` 却要调 `tools/artifact` 的 `row_split_geometry(fmt, shape)`；而本线的 `QuantFormat` 是按位宽反推每组字节数的（`group_size//2` / `group_size*(bits-4)//8`），**表达不了** PTQ1_0 的 24+2 与 PQ2_0 的 32+0。新增 `TernaryFormat(name, group_size, base_bytes_per_group, high_bytes_per_group)` 并注册两种格式。

---

## D. 重建

仓根的 `justfile` 是全部入口；本节的命令就是它实际执行的东西（`just build` / `just pack PQ2_0` /
`just patch-check` / `just e2e <artifact>` / `just bench <artifact>`）。`just config` 打印它解析出的路径。

```bash
# Linux / sm_89（4090）；3090 用 NINFER_ARCH=86
export NINFER_ROOT=/path/to/ninfer-4090
tools/verify/build.sh clean          # 首次；之后用 incremental
```

干净检出（`git clone` 出来的 v1.2.0）上跑完整条链路是一条命令：

```bash
just from-scratch PQ2_0              # patch-apply -> build -> build-tests -> ctest -> pack
```

这条链路在干净检出上核对过：`apply` 写入 45 个文件后，目录树与已应用检出逐字节一致（`diff -r` 无差异）。

不想克隆本仓时，`uv tool install` 把这条链路搬进了 wheel 构建阶段：拉取上游 → 落地同一份补丁 →
落地自检 → CMake/Ninja 编译 → 把可执行文件与打包器打进 wheel → 删除临时树。装完直接得到
`ninfer` / `ninfer-serve` / `ninfer-convert` 三个命令，其中 `ninfer-convert` 就是本节的
`tools/pack.py`。见 [`docs/uv-工具安装.md`](../docs/uv-工具安装.md)。

三元制品（改了 `pack.py` 或 `tools/artifact` 之后必须重打）：

```bash
export NINFER_TERNARY_GGUF=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf
export NINFER_TERNARY_TEMPLATE=/path/to/qwen3_8_27b_groupwise_int.ninfer
python3 tools/pack.py build out.ninfer
```

模板必须是 **groupwise-int** 那一版（`identity.weights_id == "groupwise-int"`）；`nvfp4` 版的 GDN/attention 是融合命名，不在映射表里。`pack.py` 现在会**开门见山地拦住**它并给出转换命令，而不是等 200 行后抛 `unmapped gdn object`。

> ### ⚠️ 制品身份：产物是 `folded-ternary`，模板是 `groupwise-int`
>
> **三元制品的 `identity.weights_id` 必须是 `folded-ternary`，"模板是 groupwise-int" 与 "产物是
> groupwise-int" 是两件事。** `pack.py` 从 v0.2.0 起写出的就是这个值。
>
> 理由：折叠三元的父权重与 groupwise-int **共用同一套行几何**（q/k 2048、v 6144），而引擎在规划期
> 只看得到形状、看不到权重，只能靠 identity 分辨两者对临时字节的需求。沿用模板的 `groupwise-int`
> 会让折叠三元按 groupwise 的容量规划，而它执行时还要多要一块 `[5120, T]` 的激活旋转缓冲 ——
> 那是**少留**，不是保守的多留。反过来，给 groupwise 制品按三元预留则只是浪费，但会被引擎自带的
> `gdn_input_proj_conv_{snapshot,record}` 测试判为"查询值与执行高水位不符"。
>
> 逐条容量查询的两档对照、三个方向的失效模式，以及"档案是否改变总容量"的配对实测，见
> [`docs/权重档案与容量规划.md`](../docs/权重档案与容量规划.md)。
>
> **因此：拿到本包之前打好的三元制品，必须重打。** 引擎对 `qwen3.8-27b` 只接受两种身份
> （`groupwise-int` → `GroupwiseIntW8Endpoints`、`folded-ternary` → `FoldedTernary`），其余组合在
> `resolve_weights()` 直接抛错，不会静默按错误的档案跑下去。

模板还必须是**容器 v2**。上游 `neroued/Qwen3.8-27B-NInfer` 的 `main` 在 v1.2.0 之后发布过 **v3** 制品
（提交 `51630a0c`「Publish v3 artifact」），而引擎侧 `src/artifact/reader.cpp` 与本脚本都只认 v1 / v2 ——
拿 v3 当模板会在 JSON 目录偏移上失败，报出一个与真实原因无关的 `JSONDecodeError`。`pack.py` 现在会先读
容器前缀并把版本号直接报出来。可用的 v2 修订版是 `dc370fb6295a`（「Update artifact with DFlash2
companion weights」，正好带 v1.2.0 那个 validate-only stub 要消费的 66 个 `dflash2/*` 张量）：

```bash
curl -L -o qwen3_8_27b.v2.ninfer \
  https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/dc370fb6295a/qwen3_8_27b.ninfer
```

---

## E. 必须做的验证

### E1. 折叠基旋转（可在本机一键复跑，已在 RTX 4090 上跑过）

```bash
NINFER_ROOT=/path/to/ninfer-4090 tools/verify/run_rotation_oracle.sh
```

它编译**引擎同一份真代码**（`ternary_rotation_kernels.cuh`），在真机上跑 6 个用例，再用 numpy 显式
Sylvester-Hadamard 矩阵比对，并对每个用例跑 3~5 个负控。本机结果：**6/6 PASS，负控全部分离**（rel_l2 ≈ 2.2e-3，负控 ≥ 0.97）。

> ⚠️ 这条路径**必须覆盖 T > 1**。ne[0] 连续意味着 `[k, tokens]` 激活是 **token 主序**：
> 元素 `(column, token)` 落在 `token * k + column`。T == 1 时 token 主序与行主序**完全重合**，
> 所以"只有 T=1 的用例全绿"什么都不能证明 —— 本包就修掉了独立 harness / oracle 里三处按行主序读数的历史写法（见 §F）。

### E2. 制品装配（需要真实权重时）

```bash
python3 tools/pack.py check
python3 tools/verify/check_payload_order.py <artifact.ninfer> text/token_embedding
python3 tools/verify/check_row_order.py   <artifact.ninfer> <pq2.gguf>
python3 tools/verify/check_signs.py       <artifact.ninfer> <gguf-metadata.json>
python3 tools/verify/check_assembly.py    <artifact.ninfer> <pq2.gguf>
```

### E3. 引擎侧端到端

```bash
tools/verify/loadtest.sh plain <artifact.ninfer>
NINFER_TERNARY_DUMP_EMBED=/tmp/embed tools/verify/gentest.sh plain <artifact.ninfer> "The capital of France is" 8
python3 tools/verify/check_embedding.py <artifact.ninfer> /tmp/embed.post.T8
```

`check_embedding.py` 对比"引擎实际写出的 embedding"与"按制品载荷独立解码 + 逆变换"的结果，T=1 与 T>1 都要跑。

### E4. 一致性矩阵（需要真实制品与 GPU）

```bash
NINFER_CLI=<build>/apps/ninfer tools/verify/e2e_ternary.sh <artifact.ninfer>
```

同一 prompt 下比 `--print-token-ids` 给出的**贪心 token 序列**：内核路径（`NINFER_TERNARY_MMA=0/1`）×
prefill 分块（128 / 1024）× MTP 投机必须逐字节一致，而负控（`NINFER_TERNARY_HADAMARD=0`）必须不一致。
只看"文本看起来对"是不够的 —— 关掉旋转之后输出会变成乱码，这条负控就是用来证明比对有判别力的。
本机结果：两个制品各 6 条用例，5 条正控同摘要、负控分离，`RESULT: PASS`。

### E5. 标准化跑分（需要真实制品与 GPU）

```bash
just bench <artifact.ninfer>          # 等价：NINFER_ROOT=<树> tools/bench/bench.sh <artifact>
```

固定语料 / 重复次数 / 预热 / prefill 分块，每条用例落成一张 tidy CSV，并把 GPU / 驱动 / CUDA / 引擎
修订 / 制品摘要 / `NINFER_*` 开关写进同一目录的 `manifest.txt`。suite（`standard` / `prefill` /
`decode` / `kv` / `mtp` / `graph` / `all`）见 `tools/bench/README.md`。

**没有 `manifest.txt` 的跑分不作为证据**：`ninfer_bench` 的默认语料是相对 CWD 的路径，换一个目录跑
换的就是语料，而输出里不留痕迹。跨制品对列前先确认两边的 `prefill_chunk` / `kv_dtype` /
`mtp_draft_tokens` / `weights_id` 一致。

---

## F. 本包顺手修掉的原包缺陷

原包的 `tools/` 是"当时工作树的快照"，与 `patches/` 并不同步。核过之后有四处会直接导致"验证通不过或结论错误"：

| # | 文件 | 症状 | 处理 |
|---|---|---|---|
| 1 | `tools/verify/harness/gemm_test.cu` | 调 `ternary_rowsplit_gemm_kernel` 时少传 `out_row_stride`，**编译不过**（内核在补丁后期加了该参数，harness 没跟上）| 补上参数；现在 `nvcc` 零错误零警告 |
| 2 | `tools/verify/oracle_rot.py` | `x.reshape(k, tokens)` 用 C 序 => 把 token 主序缓冲读成行主序。**T>1 的 3 个用例全部 FAIL**，而 T=1 的 3 个全 PASS —— 正是文档里"T=1 全绿 ≠ 正确"那句话的实物 | 改 `order="F"`；修完 **6/6 PASS** |
| 3 | `tools/verify/check_embedding.py` | 同一个 token 主序问题，注释还写反了（"op writes out as [hidden, T] row-major"）| 改 `order="F"` 并改正注释 |
| 4 | `tools/verify/harness/rot_test.cu` | `[]() -> const char* { cudaDeviceProp prop; ... return prop.name; }()` —— 返回局部变量地址（未定义行为，`nvcc` 报 `#1056-D`）| 把 `prop` 提到 `main` 作用域 |

另外：`tools/verify/check_*.py` 原先 `from _ternary_ref import Gguf`，但包里并没有 `_ternary_ref.py`（解码器只内联在 `pack.py` 里）。本包补了 `tools/_ternary_ref.py`，从 `pack.py` 重新导出，**解码器仍然只有一份实现**。

`.cmd` 批处理（MSVC + `vcvars64.bat` + `ninfer.exe`）换成了 Linux 的 `.sh`（`NINFER_ROOT` / `NINFER_BUILD_ROOT` / `NINFER_ARCH` 环境变量驱动）。

---

## G. 构建期开关

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `NINFER_TERNARY_HADAMARD` | 开 | `=0` 关闭折叠基变换。用于把"三元解码错"与"旋转错"分开二分；关掉后仍然跑完整前向、仍能测速，但数值无意义 |
| `NINFER_TERNARY_GDN_PERM` | 关 | `=1` 恢复 llama.cpp 的 GDN 特征置换。**本包 packer 已把 GDN 张量归一化成分组序，再置换一次会打乱 48 层** |
| `NINFER_TERNARY_MMA` | 开 | `=0` 强制 SIMT 路径，用于张量核路径 A/B 数值对照 |
| `NINFER_TERNARY_DUMP_EMBED` | 空 | 设为路径则 dump embedding 的 ids 与最终激活（跳过 CUDA 图捕获阶段）|
| `NINFER_TERNARY_TRACE_EMBED` | 空 | 设为任意值则打印 embedding 的 qtype 与形状 |
