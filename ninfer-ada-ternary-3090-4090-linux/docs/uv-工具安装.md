# 把本仓当工具用：uv tool install

本文只讲一件事：**不克隆本仓**，把 ninfer 引擎与模型转换器装成 PATH 上的命令。

装完只有三个可执行文件，没有仓库、没有构建目录、没有补丁文件需要你管：

| 命令 | 用途 | 来源 |
|---|---|---|
| `ninfer` | 单卡命令行推理 | 安装时现场编译的 CUDA 引擎 |
| `ninfer-serve` | OpenAI / Anthropic 兼容的推理服务 | 同上 |
| `ninfer-convert` | Ternary Bonsai 的 GGUF -> `.ninfer` 制品 | Python（随包）|

---

## 1. 前置条件

| 依赖 | 说明 | 自检 |
|---|---|---|
| Linux x86_64 | 引擎只有 Linux 构建 | `uname -m` |
| NVIDIA 驱动 + CUDA Toolkit | 需要 `nvcc` 编译三元内核 | `nvcc --version` |
| `cmake` >= 3.28、`ninja` | 构建系统 | `cmake --version` |
| `gcc` / `g++`（C++20） | 主机编译 | `g++ --version` |
| Rocky Linux 10 的 5 个系统包 | FFmpeg、libcurl 开发包等 | `tools/verify/install_deps_rocky10.sh check`（见 [依赖安装](依赖安装-RockyLinux10.md)）|
| `git` | 拉取上游与 xgrammar | `git --version` |
| `uv` | 安装器本身 | 见下 |

    curl -LsSf https://astral.sh/uv/install.sh | sh

安装过程要联网：拉上游源码、拉 xgrammar、下载服务端 Web UI 资源包（约 3 MiB，来自 GitHub
release）。完全离线时见 §7.3。

支持的 GPU 架构：**sm_86（RTX 3090）与 sm_89（RTX 4090）**。上游用 `FATAL_ERROR` 硬拒其它架构，
本补丁没有放宽这个范围；装错架构会在运行时以设备检查失败收场。

---

## 2. 安装

### 2.1 从 git 装（推荐）

    uv tool install git+https://github.com/<你的账号>/ninfer-ternary.git

指定分支或标签加 `@<ref>`：`git+https://.../ninfer-ternary.git@v0.3.0`。

要一并拿到模型转换能力，加上 `convert` 额外项 —— 打包器要读写张量，依赖 torch，所以它不在
默认依赖里：

    uv tool install "ninfer-ternary[convert] @ git+https://github.com/<你的账号>/ninfer-ternary.git"

### 2.2 从本地目录装

仓库已经在你手上时：

    cd ninfer-ternary
    just tool-install          # 等价于 uv tool install --force .
    just tool-install-convert  # 再加上转换器：uv tool install --force ".[convert]"

### 2.3 从 wheel 装（离线或内网）

在一台联网且依赖齐全的机器上先构建 wheel（这一步同样会编译引擎）：

    uv build --wheel .
    # dist/ninfer_ternary-0.3.0-py3-none-linux_x86_64.whl

把 wheel 拷到目标机（目标机仍需有 NVIDIA 驱动与 CUDA 运行时，但不需要编译工具链）：

    uv tool install ./ninfer_ternary-0.3.0-py3-none-linux_x86_64.whl

---

## 3. 安装过程中发生了什么

`uv tool install` 会调用本仓自带的 PEP 517 构建后端（`build_backend.py`），在 **wheel 构建阶段**
依次完成：

| 步骤 | 做什么 | 中途产物 |
|---|---|---|
| 1 | 按 `patches/manifest.json` 钉死的提交取上游 ninfer-4090（先试单提交浅取，失败退回完整克隆）| 临时源码树 |
| 2 | 落地 45 个文件的补丁，跑 20 条落地自检；不过就中止，不会产出坏 wheel | 同上 |
| 3 | CMake 配置 + Ninja 编译 `ninfer` 与 `ninfer-serve` | 临时构建目录 |
| 4 | 把两个可执行文件、上游制品读写模块、打包器与补丁打进 wheel | wheel 文件 |
| 5 | 删除整个临时树 | 无 |

所有临时物都落在**一个项目专属根目录**下：`$TMPDIR/ninfer-ternary/`（默认 `/tmp/ninfer-ternary/`），
里面再按用途分 `wheel-*` 与 `build-*`。编译子进程的 `TMPDIR` 也被指到临时树内部，所以连 nvcc 的
`tmpxft_*` 中间文件都在根里。**宿主机的工程目录里不会留下 `ninfer-*` 目录**；只有第 4 步的 wheel
被 uv 收进缓存。

> 用 `kill -9` 打断构建会跳过清理（进程被杀就没人执行删除），根目录会留下临时树；
> `just clean` 删掉整个根，并顺手清掉改名前的 `ninfer-ternary-*` / `nifer-ternary-*` 遗留目录。

本机实测（Rocky Linux 10 / RTX 4090 / 128 核）：

| 指标 | 值 |
|---|---|
| 安装总耗时 | **6 分 31 秒**（`-j 128`）|
| 生成的 wheel | **223 MiB**（`ninfer_ternary-0.3.0-py3-none-linux_x86_64.whl`）|
| 装好的工具环境 | **517 MiB**（引擎本体，两个可执行文件约 480 MiB）|
| 带 `[convert]` 时 | **5.0 GiB**（多一个 CUDA 版 torch）|
| 构建期临时根占用 | 约 **0.7 GiB**（临时源码树 + 构建目录 + nvcc 中间文件），安装结束连根目录一起消失 |

编译并行度默认取本机 CPU 核数（本机 128），可用 `NINFER_TERNARY_JOBS` 压到更小。

---

## 4. 装完自查

    ninfer --help | head -3
    ninfer-serve --help | head -3
    ninfer-convert            # 打印转换器用法

三个命令都在 `~/.local/bin`（`uv tool install` 默认的二进制目录，`uv tool dir --bin` 可查）。
如果 `ninfer` 报"找不到引擎可执行文件"，说明这次装的是"只装 Python 侧"的变体，见 §7.1。

---

## 5. 构建期环境变量

写在 `uv tool install` 前面即可：

| 变量 | 默认 | 作用 |
|---|---|---|
| `NINFER_TERNARY_ARCH` | `89` | CUDA 架构，`86` 或 `89` |
| `NINFER_TERNARY_JOBS` | CPU 核数 | 编译并行度 |
| `NINFER_TERNARY_SKIP_BUILD` | 关 | 置 `1` 只装 Python 侧，不编译引擎（wheel 退化为纯 Python）|
| `NINFER_TERNARY_KEEP_BUILD` | 关 | 置 `1` 保留临时树，编译失败时排查用 |
| `NINFER_TERNARY_TMPDIR` | 空 | 改写临时根目录，默认 `$TMPDIR/ninfer-ternary` |
| `NINFER_TERNARY_ENABLE_UI` | 上游默认（开）| 置 `0` 跳过服务端 Web UI 资源下载，`ninfer-serve` 用空 UI 桩 |
| `NINFER_TERNARY_RUN_TESTS` | 关 | 置 `1` 连引擎自带测试套件一起编译并 `ctest` |
| `NINFER_TERNARY_TARGET_REPO` | manifest 里的上游地址 | 换成内网镜像或本地检出 |
| `NINFER_TERNARY_SOURCE` | 空 | 直接复用一棵已有检出，跳过克隆（该目录不会被删除）|
| `NINFER_TERNARY_BUILD_ROOT` | 系统临时目录 | 临时树的父目录 |

例如装 3090 用的 sm_86 版本：

    NINFER_TERNARY_ARCH=86 NINFER_TERNARY_JOBS=32 uv tool install --force .

运行期（不是构建期）还会用到：`NINFER_ENGINE_BIN`（指向引擎可执行文件目录）、
`NINFER_ROOT`（转换器找上游制品模块）、`NINFER_TERNARY_TEMPLATE` / `NINFER_TERNARY_GGUF`。

---

## 6. 升级与卸载

    uv tool list                        # 看装了什么
    uv tool upgrade ninfer-ternary      # 从原来源升级（会重新编译）
    uv tool uninstall ninfer-ternary    # 卸载
    uv tool install --force --no-cache .   # 强制重编，忽略 wheel 缓存

---

## 7. 常见场景

### 7.1 只想要转换器

默认安装不带转换器 —— 它要 torch（约 2 GB），跑推理的人不需要：

    just tool-install-convert      # 等价于 uv tool install --force ".[convert]"

连 CUDA 工具链都没有（编译根本跑不起来）时，再加"只装 Python 侧"：

    NINFER_TERNARY_SKIP_BUILD=1 uv tool install --force ".[convert]"

这种安装里 `ninfer` / `ninfer-serve` 会提示找不到引擎；`ninfer-convert` 本身可用，但没有
随包的上游制品模块，要指向一棵 ninfer 检出：

    NINFER_ROOT=/path/to/ninfer-4090 ninfer-convert --template ... --gguf ... check

想事后补上引擎，直接用默认方式重装即可（编译发生在安装时，没有"事后编译"这一步）。

### 7.2 内网 / 镜像

    NINFER_TERNARY_TARGET_REPO=ssh://git@内网/ninfer-4090.git uv tool install --force .

仓库里已有打好补丁的检出时，也可以直接复用它，省掉克隆：

    NINFER_TERNARY_SOURCE=/path/to/patched/ninfer-4090 uv tool install --force .

### 7.3 完全离线

在联网机器上 `uv build --wheel .` 得到 wheel，拷到目标机 `uv tool install <wheel>`。
注意 wheel 里已经包含编译好的引擎，目标机只需要 NVIDIA 驱动与 CUDA 运行时。

### 7.4 编译失败要看现场

    NINFER_TERNARY_KEEP_BUILD=1 uv tool install --force . 2>&1 | tee install.log

失败时日志里会打印临时目录路径（`临时目录: /tmp/ninfer-ternary/build-xxxx`），构建日志与
CMake 缓存都还在那里。

---

## 8. 排错

| 症状 | 原因 | 处置 |
|---|---|---|
| `error: no such command` / 找不到 `just` | 本机没装 just | 直接用 `uv tool install --force .` |
| 构建卡在 `downloading WebUI release` | GitHub release 下载慢 | `NINFER_TERNARY_ENABLE_UI=0` 重装 |
| `nvcc: command not found` | 没装 CUDA Toolkit 或不在 PATH | 装 CUDA Toolkit；或先装"只装 Python 侧"版本 |
| `CMake Error ... libavformat ... not found` | 缺 Rocky 10 的 5 个系统包 | `tools/verify/install_deps_rocky10.sh install` |
| 落地自检不过，安装中止 | 上游提交与补丁清单不匹配 | 确认 `NINFER_TERNARY_TARGET_REPO` 指向的是同一个上游 |
| `ninfer` 提示找不到引擎 | 装的是 `SKIP_BUILD` 变体 | 用默认方式重装 |
| `ninfer-convert` 提示缺少 torch | 安装时没带 `convert` 额外项 | `uv tool install --force ".[convert]"` |
| 运行时设备检查失败 | 引擎架构与显卡不符 | `NINFER_TERNARY_ARCH=86`（3090）重装 |
| 临时根里留下 `wheel-*` / `build-*` | 构建被 `kill -9` 打断 | `just clean`（删掉整个 `$TMPDIR/ninfer-ternary`）|

---

## 9. 与"克隆仓库跑 just"的分工

| 你要做的事 | 用哪个 |
|---|---|
| 装一个能跑推理的引擎 | `uv tool install`（本文）|
| 转换模型 | `ninfer-convert`（装完即用）|
| 改补丁、跑 ctest、跑端到端矩阵、跑分 | 克隆仓库，用 `just`（见 [README](../README.md)）|
| 复现本仓的验证结论 | 克隆仓库，`just from-scratch PQ2_0` |
