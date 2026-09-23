# ninfer-ada-ternary-4060

把 [NInfer](https://github.com/UDPSendToFailed/ninfer-4090)（Apache-2.0）的**三元**能力
（Ternary Bonsai 2 27B）落到 **RTX 4060 8G（sm_89）** 上——当前主要修改目标；
sm_86 / sm_89（3090 / 4090）为兼容与对照平台。

引擎源码已在树内（合入上游 v1.2.0 + 三元补丁），`uv tool install` 时树内 cmake 编译后打进 wheel。
装完你得到三个可执行文件：

| 命令 | 用途 |
|---|---|
| `ninfer` | 单卡命令行推理 |
| `ninfer-serve` | OpenAI / Anthropic 兼容的推理服务 |
| `ninfer-convert` | 把 Ternary Bonsai 的 GGUF 转成 ninfer 制品（`.ninfer`）|

本仓不发布任何模型权重，也不发布由权重派生的 `.ninfer` 制品。

---

## RTX 4060 主线（当前目标）

本仓当前的主要修改目标是 **RTX 4060 8G（sm_89）**，主力格式 **PTQ1_0**。
与 4090 同为 Ada sm_89，代码层完全兼容，差异只在显存容量与算力档位。
完整决策、阶段状态与实测记录见 [docs/4060-开发跟踪.md](docs/4060-开发跟踪.md)。

### 一键构建与打包

    ./build_4060.sh            # Linux / WSL：增量构建 → build_4060/apps/ninfer
    build_4060.bat             # Windows 等价入口
    pack_ptq1_4060.bat         # 打包 PTQ1_0 制品（保留 mtp/vision，裁 dflash2）

低内存（7.6 GiB）构建机默认 `NINFER_JOBS=1`，防单文件 CUDA 编译 OOM。

### 生成的制品（2026-09-23 实测）

| 制品 | 大小 | 说明 |
|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0-text.ninfer` | 7,047,407,628 B（6.563 GiB） | 1126 对象 = frontend 6 + text 775 + mtp 12 + vision 333，无 dflash2 |
| `build_4060/apps/ninfer` | 235 MB | sm_89 可执行（`ninfer-serve` 246 MB） |

制品绑定结论：引擎把 mtp/vision 当**必选存在项**，唯一可裁的是 dflash2；
早期裁掉 mtp/vision 的 text-only 制品启动即报缺对象，已弃。

### 正确性冒烟（4060 8G，全过）

| 用例 | 结果 |
|---|---|
| 贪心 `17 * 23` | **391** |
| `MMA=1` vs `MMA=0` | 逐字节一致（hash `b327e287e849`） |
| `HADAMARD=0` 负控 | 乱码，与正控分离 |
| 装载显存 | 权重 5.52 GiB，8G 卡内可起跑 |

建议起跑参数：`--kv-dtype rk4v4-e8 --max-context 3584 --prefill-chunk 1024`。
MTP 投机本期不支持（8G 卡预留 924 MiB 不足）。

### 性能基线（4060，PTQ1_0，仅记录不调优）

> 2026-09-24 起 decode 走 **PTQ1_0 SIMD 4-trit 解码 warp-per-row GEMV 快路径**（`NINFER_TERNARY_PTQ1_GEMV=1`），
> 对比全程参考解码可回退：`NINFER_TERNARY_PTQ1_GEMV=0`。token 序列与参考逐 token 一致。

| 项 | 实测 |
|---|---|
| **decode（新快路径）** | **8.33 tok/s**（greedy / rk4v4-e8 / ctx2048 稳态，192 tok） |
| decode（参考解码回退） | **3.36 tok/s**（greedy / rk4v4-e8 / ctx2048 稳态） |
| prefill | **9.09 tok/s**（同上，28 tok prompt） |
| 参照：4090 PTQ1_0 | 15.8 tok/s decode |
| 参照：llama.cpp 同机同文件 | **25 t/s**，差距 ≈ ×3.0（立案前 ×7.4） |

decode 快路径收益 ≈ **×2.48**（3.36 → 8.33），源自 SIMD 4-trit 解码 + warp-per-row 结构改写，
实现与验证见开发文档 §9.3。剩余 ×3.0 差距（llama.cpp 25 t/s 对照）仍是二期首要优化项，
跟踪见开发文档 §9.1。

差距已立案为二期首要优化项（PTQ1_0 base-3 慢路径、缺 GEMV/MMA 快指令），
跟踪见开发文档 §9.1。两种格式的对比结论见开发文档 §8。

---

## 一分钟开始

前提：Linux x86_64、NVIDIA 驱动与 CUDA 工具链、`git`、`cmake`、`ninja`、`gcc`、`uv`。
Rocky Linux 10 上可以用仓内脚本一次装齐（见 [依赖安装](docs/依赖安装-RockyLinux10.md)）：

    tools/verify/install_deps_rocky10.sh install

还没有 `uv` 的话：

    curl -LsSf https://astral.sh/uv/install.sh | sh

安装过程需要联网：除了拉上游源码，CMake 配置阶段还会取 xgrammar 与服务端 Web UI 两个第三方包。
离线环境请在一台联网机器上先 `uv build --wheel`，再把 wheel 拷过去装（见教程）。

装推理引擎（会现场编译 CUDA 引擎，本机实测 6 分半）：

    uv tool install git+https://github.com/<你的账号>/ninfer-ternary.git

本地已有本仓时直接指目录，效果相同：

    just tool-install          # 等价于 uv tool install --force .

装完立刻可用，不需要仓库、也不需要构建目录：

    ninfer --help
    ninfer-serve --help

要转换模型再加一个额外项：打包器要读写张量、依赖 torch，所以转换能力单独放：

    uv tool install "ninfer-ternary[convert] @ git+https://github.com/<你的账号>/ninfer-ternary.git"
    just tool-install-convert  # 本地目录的等价写法：uv tool install ".[convert]"

    ninfer-convert             # 打印转换器用法

细节、环境变量、离线安装与排错见 [把本仓当工具用](docs/uv-工具安装.md)。

---

## 转换一个模型

`ninfer-convert` 来自上面带 `[convert]` 额外项的安装；只装了引擎的安装会在调用时
明确提示补装，不会让你对着 traceback 猜。

转换需要两样输入：

| 输入 | 说明 |
|---|---|
| GGUF | `Ternary-Bonsai-2-27B-PQ2_0.gguf` 或 `-PTQ1_0.gguf`（HuggingFace 上的 Ternary Bonsai 2 27B）|
| 模板 | 一个 **groupwise-int** 的 qwen3.8-27b ninfer 制品。模板提供视觉塔、MTP 头等"借用"张量与对象清单，不是可有可无的参考文件 |

    ninfer-convert \
      --template /data/Ternary-Bonsai-2-27B-ninfer/template/qwen3_8_27b.v2.ninfer \
      --gguf     /data/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
      build      /data/Ternary-Bonsai-2-27B-ninfer/Ternary-Bonsai-2-27B-PQ2_0.ninfer

写盘之前会先做只读自检（几何、解码往返、张量映射），确认没问题再落盘：

    ninfer-convert --template <模板> --gguf <GGUF> check

两个路径没有内置默认值，必须显式给出或走环境变量 `NINFER_TERNARY_TEMPLATE` /
`NINFER_TERNARY_GGUF`。本机实测产物：

| 制品 | 大小 | 说明 |
|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0.ninfer` | 10,533,732,876 B | 2 bit 权码，精度更高、解码更快 |
| `Ternary-Bonsai-2-27B-PTQ1_0.ninfer` | 9,274,212,876 B | 三进制 + 高位平面，更省空间 |

---

## 跑推理

命令行单卡：

    ninfer /data/Ternary-Bonsai-2-27B-ninfer/Ternary-Bonsai-2-27B-PQ2_0.ninfer \
      --prompt "17 * 23 等于多少？" --max-context 4096 --max-new 256

起服务并请求：

    ninfer-serve /data/Ternary-Bonsai-2-27B-ninfer/Ternary-Bonsai-2-27B-PQ2_0.ninfer \
      --host 127.0.0.1 --port 8080 --max-context 8192

    curl http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"你好"}]}'

常用的几组参数（完整列表见 `--help`）：

| 参数 | 作用 |
|---|---|
| `--max-context` / `--prefill-chunk` | 上下文长度与预填充分块（决定显存里的工作区大小）|
| `--kv-dtype bf16\|int8\|rk8v4\|rk4v4\|rk4v4-e8\|rk2v4-e8` | KV 缓存精度，越小越省显存 |
| `--spec mtp --draft-tokens 4` | MTP 投机解码，输出与不开投机逐字节一致 |
| `--no-cuda-graph` | 关掉 CUDA Graph（排查问题或显存吃紧时用）|

本机（RTX 4090，PQ2_0）实测：prefill 512 token **267.2 ± 18.4 t/s**，解码 **50.8 ± 4.3 t/s**。
跑分脚本与口径见 [tools/bench/README.md](tools/bench/README.md)。

---

## 常见问题

**装完 `ninfer` 报"找不到引擎可执行文件"。**
这次安装是"只装 Python 侧"的（`NINFER_TERNARY_SKIP_BUILD=1`），wheel 里没有引擎。
用 `just tool-install` 重装；或把 `NINFER_ENGINE_BIN` 指向已有的构建产物目录。

**没有 CUDA 工具链，只想要转换器。**
`just tool-install-light`（即 `NINFER_TERNARY_SKIP_BUILD=1 uv tool install --force .`）。
这种安装没有随包的上游制品模块，转换时要用 `NINFER_ROOT=<ninfer 检出>` 指一个过去。

**模板从哪里来？**
模板是 qwen3.8-27b 的 **groupwise-int** ninfer 制品，与目标制品同为容器 v2。转换器会先读它的
`identity.weights_id` 校验，不是 `groupwise-int` 会直接拒绝并说明原因 —— 用成别家的量化（例如
nvfp4）会在后面以张量名对不上的形式失败。

**上游仓库在内网拉不到。**
`NINFER_TERNARY_TARGET_REPO=<镜像地址或本地检出> uv tool install --force .`。

**占多少显存？怎么估算？**
工作区容量随 `min(max_context, prefill_chunk)` 线性增长，与制品档案无关；实测与推导见
[权重档案与容量规划](docs/权重档案与容量规划.md)。

---

## 在仓库里开发

仓根的 `justfile` 是全部入口。`just` 列出配方，`just config` 打印它解析出来的路径。

    just check          # ruff / mypy / pytest
    just deps           # 构建依赖自检
    just build          # 增量构建 sm_89（just build 86 编 sm_86）
    just build-tests && just ctest
    just build-engine   # 拉上游 -> 打补丁 -> 自检 -> 编译，临时树自动清理
    just oracle         # 旋转内核 vs numpy FP64
    just e2e <制品>     # 端到端一致性矩阵（结果在 out/e2e-<制品名>）
    just bench <制品>   # 标准化跑分（结果在 out/bench-<制品名>-<时间戳>）
    just pack PQ2_0     # 打包三元制品
    just clean          # 清掉构建目录、字节码缓存与临时根

补丁侧的子命令（`manifest` / `status` / `apply` / `check` / `export`）走
`uv run python -m ninfer_ternary`，不随 `uv tool install` 安装 —— 它们只在开发与验证时需要。

### 仓里有什么

    patches/            45 个文件的整文件快照 + 清单摘要（改动清单见 patches/README-改动说明.md）
    tools/pack.py       GGUF -> .ninfer 打包器（ninfer-convert 的本体）
    tools/verify/       oracle、端到端矩阵、依赖安装、落地自检
    tools/bench/        固定语料/重复/热身的标准化跑分
    build_backend.py    uv tool install 时拉取、打补丁、编译、清理
    docs/               移植报告、权重档案、依赖安装、本工具安装

主要文档：

| 文档 | 内容 |
|---|---|
| [把本仓当工具用](docs/uv-工具安装.md) | `uv tool install` 全流程、环境变量、离线安装、排错 |
| [移植报告](docs/移植报告-ninfer-4090.md) | 判定依据、实测证据、未验证部分 |
| [权重档案与容量规划](docs/权重档案与容量规划.md) | 制品档案改变了什么、容量查询逐条对照 |
| [依赖安装](docs/依赖安装-RockyLinux10.md) | Rocky Linux 10 缺失库清单与安装命令 |
| [改动说明](README-改动说明.md) | 45 个文件的改动清单、与上游的刻意差异 |
| [4060 开发跟踪](docs/4060-开发跟踪.md) | 4060 主线：阶段状态、冒烟/性能实测、4060↔4090 兼容性与 PQ2_0↔PTQ1_0 对比 |

---

## 验证状态（本机实测，RTX 4090 / Rocky Linux 10）

| 项 | 结果 |
|---|---|
| 构建 | sm_89 与 sm_86 各 exit 0，三元内核在两个架构下都有原生 cubin |
| 引擎自带测试 | `ctest` 84/84 通过 |
| 端到端 | 两种格式都装载并答对 `17 * 23`；`MMA=1` 与 `MMA=0` 逐字节一致；关掉折叠基旋转即崩坏 |
| 一致性矩阵 | 内核路径 × 分块 × 两种格式，10 次正控同摘要，负控分离 |
| MTP 投机 | 输出与无投机逐字节一致，接受率 74-77%（draft 4）|
| 长上下文 | 2685 与 11043 token 的 prompt 全部同摘要；三元 MMA prefill 约为 SIMT 的 4.2-4.4 倍 |
| 干净检出可复现 | `git clone` v1.2.0 -> 打补丁 -> `diff -r` 无差异；全量重编 726/726 exit 0，`ctest` 84/84 |
| **`uv tool install` 一条命令** | 现场拉取 v1.2.0 -> 落地 45 文件 -> 自检 20/20 -> 编译 -> 打成 223 MiB wheel -> **临时根整个删除**（`/tmp` 不留文件也不留空目录），全程 **6 分 31 秒**；装好的 `ninfer` 直接答对 `17 * 23` |

---

## 许可与来源

本仓是 **Apache-2.0** 许可，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

它是 NInfer（Apache-2.0）派生作品的适配层：三元改动来自
[ninfer-ada-ternary](https://www.modelscope.cn/shensanshu/ninfer-ada-ternary.git)，
目标树是 [ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) v1.2.0
（提交 `5c60b7c9`）。上游只支持 sm_86 / sm_89，本补丁没有收窄或放宽这个范围。

模型权重不在本仓分发，其权利归原作者（PrismML / Qwen 体系）所有。
