# 4060 / ninfer × KVMem 评估

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-23 |
| 目标树 | `D:\Downloads\dev\LLM_base\llm_opt\llm_infer\ninfer\ninfer-ada-ternary\ninfer-ternary\ninfer-ada-ternary-3090-4090-linux` |
| 参照树 | `D:\Downloads\dev\LLM_base\llm_opt\llm_infer\llamacpp\kvmem-llama.cpp` |
| 性质 | 只读评估与借鉴方案，未改代码 |
| 相关 | `docs/4060-开发跟踪.md`（既有 4060 线） |

---

## 1. 结论（先看这节）

1. **同题不同解。** KVMem 与 ninfer 都在省「长上下文 KV 的 GPU 显存」；ninfer 用**激进 KV 量化 + Paged 全量驻留**，KVMem 用**有界工作集 + host 分层 + 按 query 检索块**。量化装得下时，KVMem 边际收益小；**逻辑上下文超过 `kv_capacity` / 卡上算力吃紧**时，KVMem 才有独特价值（省的不只是字节，还有 **attention 不随历史线性涨**）。
2. **可借鉴，不可照搬。** 可整段拿走的是 `kvmem/` 纯 host 库（选块、plan/diff、分层、打分）；不能拿的是 `llama_memory_*`、ggml 图 hook、`llama-kvmem-stagein.cu`、query-replay 的 llama 状态机。ninfer 侧要补：**pre-RoPE Q/K 捕获**、**有界可见页表（或 cold 页换入）**、**块级 D2H/H2D backend**。
3. **最大边界：** ninfer 文档把 **KV offload / 跨卡** 写成 non-goal；引入 KVMem 必须先改产品边界。质量上 KVMem 是**稀疏近似**，与 ninfer「与 baseline 逐字一致」的契约冲突，须另开 needle/agent 质量门。
4. **Hybrid 硬约束：** 只稀疏 **softmax attention 半边**；**GDN/ReplaySSM 必须看满每个 token**（KVMem hybrid 同样只稀疏 attn）。MTP / DFlash 的 growing pool 必须与主 KV **同一套 budget/换页纪律**（对应 KVMem MTP 方案 B follower 池），否则主 KV 省了、投机池按全长涨回去。
5. **推荐路径：** 分四步——`P1` 冷页 host spill（仍全量 attention，非目标先行）→ `P2` 有界窗口（sink+recent / H2O，差分换页）→ `P3` mean-K 检索 + budget/gen_reserve + MTP/DFlash 对齐 → `P4` 工具轮 query/frontier 跳过。**P1/P2 改动面小、可先验证速度与正确性；检索是第二步。**

---

## 2. KVMem（kvmem-llama.cpp）是什么

- 论文：[KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852)（`chai2026kvmem`）。
- 形态：llama.cpp **子模块 + 薄补丁**；`kvmem/` 零依赖 llama 头；`src/adapter/` 实现 `llama_memory_i`。
- 主张：256K 查询下仅保留 **~32K GPU 活跃上下文** 接近无损（LongMemEval-S 85.6% vs 86.6%；AgentLongBench 60.9% vs 59.5%）。
- 与 adaptive KV streaming 的区别：streaming 仍对**全历史**做 attention + 逐层预取；KVMem **限制参与 attention 的 KV 量**，历史在 host，按 query 装入 `budget + gen_reserve`。

### 2.1 创新点摘要

| 层 | 要点 |
|---|---|
| 问题 | 消费级 16GiB 上 million-token agent 工作区 |
| 算法 | query 条件块检索；mean-K（pre-RoPE F32）廉价索引；Retrieval 可复活 / H2O 热度 / Recency；sink+top-k |
| 系统 | 有界**块槽池**；**Reselect = KvMemPlan 差分**（skip 零拷贝）；**不改 FA**；cell 保留**原始 pos**，冷块 packed memcpy 免 re-RoPE |
| 正确性 | Query replay：六条执行路径、可证明 skip、Q/检查点复用；工具 prefill 中位 14.9s→6.5s |
| 异构 | 只稀疏 attn；GDN 全序列；MTP follower **共用 slot 池**（256K draft KV ~1GiB→~1.5MiB） |
| 工程 | 三 hook 补丁；packed 换出 / 32MiB slab / 异步 harvest / 进程内 prefix cache |

### 2.2 可移植边界

**可原样移植（`kvmem/`）**

- `KvMemStore`：块表、`pick_topk`、sink/recent、`KvMemPlan`/`KvMemRemap`、`set_selection`。
- `KvMemRuntime`：reselect 编排，顺序 **spill_outgoing → 释放逻辑页 → admit_incoming → finish**。
- 分层：`pinned_kv_tier`（CPU slab）、`raw_kv_store`（mean-K + packed K/V）、`nvme_kv_tier`（POSIX，Windows 需 shim）。
- Backend 抽象（仅 4 个虚函数）：

```text
alloc_gpu_slot / free_gpu_slot
copy_block_to_host / copy_block_from_host
```

- 打分：MeanK / PerToken / SubBlockMeanK；host 测试无 GPU。

**需在 ninfer 重写的适配面**

1. pre-RoPE **Q/K（可选 V）** 捕获 hook。  
2. **有界可见集**：仅把选中块挂进 FA 页表，或接受「host 全量、GPU 仅 budget」。  
3. `KvMemBackend` → paged pool 的块 D2H/H2D（可对齐 `kv_paged_staging` gather/scatter）。  
4. KV 量化 pack/unpack 对齐 ninfer dtype（int8 / rk* / e8 lattice），**不要**搬 ggml q8_0/q5_0 布局。

**不要搬**

- `llama-memory-kvmem*`、图 capture、`seq_rm_logical`、hole-purge 补丁。  
- `llama-kvmem-stagein.cu`（除非只抄 32MiB slab 思路）。  
- llama 版 query-replay 的 recurrent/MTP payload 与 server executor（概念可抄，状态机重写）。  
- M-RoPE / mm 路径、Windows 上未 shim 的 NVMe tier。

### 2.3 关键旋钮（16GiB 配方）

| 旋钮 | 16GiB 默认 |
|---|---|
| `block_tokens` | 128 |
| `budget`（检索驻留） | IQ3 36864 / IQ4 32768 |
| `gen_reserve`（生成预留） | 16384 / 12288 |
| GPU 池 | `budget + gen_reserve` |
| method | retrieval；MeanK；query last 64 / cap 512 / policy user |
| replay | auto |
| 主 KV | q8_0 / q5_0；MTP draft f16 |
| ctx | 262144 |

已知 v1 限制：检索 pin 后单轮生成 ≤ `gen_reserve`（含 thinking）；文档规划 gen_reserve 内 ring、本 port **NVMe 未实现**。

---

## 3. ninfer（本目标树）现状

### 3.1 身份与能力

- 包：`ninfer-ternary`（Ternary Bonsai 2 27B → sm_86/89）；引擎自上游钉提交拉取 + 三元补丁。
- 命令：`ninfer` / `ninfer-serve` / `ninfer-convert`。
- KV：`--kv-dtype bf16|int8|rk8v4|rk4v4|rk4v4-e8|rk2v4-e8|...`。
- 量化收益（上游 release）：约 **260→100 B/tok**；24GB 上宣称 **~350K** 驻留；相对 FP32 cosine ~96–99.8%，needle 有 1M 报告。
- 引擎侧：Paged KV、`kv_capacity` 自动规划（headroom ~1GiB）、MTP/DFlash 独立 growing pool、GDN/ReplaySSM、prefix/frontier checkpoint、可选 WDDM evictable budget。
- 并发：文档目标 `max_concurrency=2..8`，paged 与 admission 记账完整。

### 3.2 关键结构（只读索引）

| 主题 | 路径（相对目标树） |
|---|---|
| Paged 设计 | `docs/maintainer/paged-kv-cache.md` |
| 并发/frontier | `docs/maintainer/concurrent-inference-architecture.md` |
| Paged 视图 | `src/core/paged_kv_cache.h`（`PagedKVLayerView`、page=64、block_table） |
| 容量 | `src/runtime/engine/kv_capacity.{h,cpp}` |
| 循环窗 | `src/core/cyclic_kv_cache.*`（SWA 等） |
| 追加/前缀 | `kv_cache_append*`、`include/ninfer/ops/kv_cache_append_prefix.h` |
| Staging | `src/ops/kernel/kv_paged_staging.cuh`（gather/scatter page-major） |
| Attention | `softmax_attention` / `sliding_window_attention` 等 op |
| GDN | `gated_delta_net`、`gdn_replay` |
| 投机池 | `paged-kv-cache.md` §3：Main / MTP / DFlash Full-Context pools |
| 量化说明 | `RELEASE_NOTES_0.9.0.md`、README `--kv-dtype` |
| 4060 跟踪 | `docs/4060-开发跟踪.md`（如存在，与本评估并列） |

### 3.3 与「省显存」相关的已有杠杆

1. **更小的 bytes/tok**（量化）——已是主力。  
2. **`kv_capacity` / `max_context` 收缩**——装不下就拒/缩。  
3. **Paged 非连续**——并发碎片友好，**不是**把历史挪出 GPU。  
4. **WDDM**——Windows 桌面抢占，不是 KV 分层。  
5. **明确 non-goal：** request preempt、**KV offload**、跨 GPU；`arbitrary LCP` 也非目标（prefix 仅完整 checkpoint）。

---

## 4. 对照：同题异构

| 维度 | kvmem-llama.cpp | ninfer（本树） |
|---|---|---|
| 省显存主手段 | host 仓 + **检索**有界窗 + 中等量化 | **强量化** + 池容量规划，**全量驻留** |
| Attention | 只扫 budget 窗口 | 驻留 frontier 内**稠密全扫** |
| 超容量 | 换页/检索 | 缩 `max-context` 或失败 |
| 页/块 | 逻辑 128-token 块 + GPU 槽 | 物理 page=64 + block_table |
| 位置 | cell **原 pos**，免 re-RoPE | page 寻址 + 各 op 的 pos/frontier 语义 |
| 质量契约 | 稀疏近似，检索 GO/NO-GO | 常与 dense baseline 对齐 / 逐字（MTP 等） |
| 并发 | 基本单序列 | 2–8 活跃请求设计 |
| GDN | hybrid 只稀疏 attn | 全序列 GDN + ReplaySSM |
| 投机 | MTP follower 同槽池 | 独立 MTP/DFlash pool |
| 非目标 | 不改 FA、少改 llama 核心 | **不做 KV offload**、不做 arbitrary LCP |

---

## 5. 可借鉴方案（按侵入度）

### 阶段总览

```text
权重（三元）           已很省
KV 量化 (int8/rk*)     已很省
若 logical_ctx * B/tok > kv_capacity:
  P1  冷页 host spill + 仍全量 attention     ← 改 non-goal，不改数学
  P2  有界窗口 sink+recent/H2O + plan 差分   ← 省显存+省算力
  P3  mean-K 检索 + budget/gen_reserve       ← 完整 KVMem 核心
  P4  工具轮 query/frontier 跳过             ← 对齐 query replay 概念
MTP/DFlash 池与主 KV 同一预算纪律（始终）
GDN 不做块稀疏（始终）
```

### P1 — 冷页换出到 host（仍全量 attention）

- **做什么：** 超出工作集的 paged 组 D2H 到 host；需要时 H2D；逻辑 frontier 与 attention 语义不变。  
- **接口：** 在 `KvMemBackend` 意义上实现块拷贝；驱逐可接现有 page 生命周期，**不必**先上检索。  
- **为何先做：** 改的是存储边界，不改 softmax 数学；可先验证正确性与 PCIe 成本（对照 KVMem streaming 批评：过长上下文 PCIe 会涨）。  
- **风险：** 改 `paged-kv-cache.md` non-goals；serving admission / capacity solver 要计入 host tier 字节。

### P2 — 有界窗口（无内容检索）

- **做什么：** 固定只保留 sink + recent（或 H2O 热度）；中间块按 P1 下放；**可见集 = GPU 上的页**。  
- **对应 KVMem：** `Recency` / `H2O` 路径与 `pick_topk_blocks` 的子集。  
- **收益：** decode 复杂度与显存不再随 `max-context` 线性涨。  
- **风险：** 丢中间事实；需 needle + 工具回放门禁。  
- **实现要点：**  
  - 选块仍可用 `kvmem/` store（只开 sink+recent+score 热度）；  
  - `apply` 改为 **page 维度 plan diff**（resident 页零拷贝）；  
  - 与 `kv_capacity` 记账并存：工作集字节 ≤ 预算，冷页不占 pool 时要单独配额。

### P3 — 检索式工作集（KVMem 核心）

- **做什么：** pre-RoPE mean-K 索引；query 选 top-k；`budget + gen_reserve`；pin 后新 token 只吃 reserve。  
- **KVMem 移植清单：**

| # | 项 | 来源 | ninfer 落点 |
|---|---|---|---|
| 1 | `KvMemStore` + tests | `kvmem/include|src/host` | 可 vendor 或 submodule |
| 2 | `KvMemRuntime` 编排顺序 | `kvmem_runtime.cpp` | 引擎 reselect 入口 |
| 3 | `KvMemBackend` | 4 虚函数 | 包一层 paged gather/scatter 或 slab H2D |
| 4 | Q/K 捕获 | llama graph hook | attn 前/pre-RoPE 的 tensor 读出（新 hook） |
| 5 | mean-K 写入 | harvest 线程 | prefill 块满 + decode 块满 |
| 6 | 量化 packed 换页 | stage-in/out | 用 **ninfer dtype** 布局，自写 pack/unpack |
| 7 | 可见页表 | `seq_rm` + 占用 cells | **仅挂选中页**；或 compact 后仍用原 pos（推荐原 pos） |
| 8 | MTP/DFlash | 方案 B follower | 同 block id / 同预算；禁止 draft 按 `-c` 涨 |
| 9 | gen_reserve | 双分区 | 防检索挤生成；可后续做 ring |

- **不要在 P3 做：** continuous batching × 检索（KVMem 自身也未完成）；改 GDN；改 FA kernel 读 page-index 以外的「稀疏数学」（KVMem 刻意用稠密窗规避）。

### P4 — Query / 轮次跳过

- 概念：工具续接复用 Q 特征与 checkpoint，**能证明不依赖失效则跳过重算**。  
- ninfer 已有 **frontier + turn checkpoint + prefix append**，比 llama 侧更接近；缺的是「检索改变可见历史后的依赖失效证明」。  
- 建议：先做「**选择未变 → 全保留** / **同 user 问题 → 单遍追加**」两条，完整六路径状态机后置。

### 与 4060 线关系

- 若 `docs/4060-开发跟踪.md` 已跟踪权重 offload/多卡，本评估补的是 **KV 层**：先 P1/P2，与「更大 `kv_capacity`」和量化路线并行，而不是替代。

---

## 6. 风险与门禁

| 风险 | 说明 | 门禁 |
|---|---|---|
| 质量 | 稀疏 ≠ 全量可见 | needle / 多文件事实 / 256K 工具回放；与 dense 对照，不只看单条 logits |
| 速度 | reselect + H2D 一次尖峰 | `retr_ms` 类分桶；批拷；禁止 per-token sync |
| PCIe | 长窗 streaming 重蹈覆辙 | 控制 host 流量；pin 后 decode 少换页 |
| GDN | 不能块稀疏 | ReplaySSM/Record-Fold 回归 |
| 投机 | draft 池单独涨 | MTP/DFlash 字节进同一报表 |
| 并发 | 多请求共享 pool | 每请求独立工作集记账，勿全局单窗 |
| 正确性 | 页状态×frontier×graph | 与现网 baseline 逐字/logits；P1 可做 bit 级 |
| 平台 | NVMe tier POSIX | 本树 Linux 包为主；Windows 另议 |

---

## 7. 建议决策

| 场景 | 建议 |
|---|---|
| 24G/4090 上量化后 `rk*` 已能装目标 ctx，且只需 dense 正确性 | **不必上 KVMem**；优先调 `kv-dtype` + `kv_capacity` |
| 4060 16G 级、要 256K agent 长程、decode 不能随历史变慢 | **值得做 P2→P3**；KVMem 路线互补于量化 |
| 只想先降低「装不下」的硬失败 | **P1 host spill** 最小步；改文档 non-goal 即可 |
| 要保「与 baseline 一致」的 serving 承诺 | 检索模式必须 **opt-in**，默认仍 dense/全量驻留 |

**一句话结论：**  
ninfer 已在「存得更小」上很强；KVMem 值得借的是「**装不下之后如何有界地算**」——先抄 **差分换页 + budget/gen_reserve + 只稀疏 attention 层 + MTP/DFlash 同预算**，mean-K 检索与 query 跳过放第二步；**GDN 不动，质量门与 dense 契约分开。**

---

## 8. 参考路径

**KVMem 树**

- `README.md`、`docs/architecture.md`、`docs/modification-plan.md`
- `docs/kvmem-mtp-plan.md`、`docs/query-replay-implementation-report-2026-09-14.md`
- `docs/prefill-harvest-optimization.md`、`docs/retrieval-stagein-optimization.md`
- `docs/packed-kv-prefix-cache-plan.md`、`kvmem/include/kvmem/*`、`src/adapter/*`

**ninfer 树（本目录）**

- `docs/maintainer/paged-kv-cache.md`
- `docs/maintainer/concurrent-inference-architecture.md`
- `src/core/paged_kv_cache.h`、`src/runtime/engine/kv_capacity.*`
- `src/ops/kernel/kv_paged_staging.cuh`、`RELEASE_NOTES_0.9.0.md`
- `docs/4060-开发跟踪.md`（若存在）

**论文**

- KVMem: arXiv:2609.04852
