# Rocky Linux 10 构建依赖安装说明

本文给出在 Rocky Linux 10 上为 ninfer-4090（含三元移植补丁）补齐缺失系统库的命令，并说明每一项依赖的判定依据。

判定依据是 ninfer-4090 自身的 `CMakeLists.txt` 与同仓 `Dockerfile`，不是经验推测；环境实测见第 5 节。

## 1. 结论：需要安装的只有 5 个包

```bash
# EPEL 提供 ffmpeg-free 系列，Rocky 10 官方仓库不含 FFmpeg 开发包
sudo dnf install -y epel-release

sudo dnf install -y --setopt=install_weak_deps=False \
  cmake ninja-build pkgconf-pkg-config \
  libavformat-free-devel libavcodec-free-devel libavutil-free-devel libswscale-free-devel \
  libcurl-devel
```

其中 `ninja-build` 与 `pkgconf-pkg-config` 在本机已存在，列出来是为了命令在全新机器上同样可用（dnf 对已装包是幂等的）。

`xgrammar`、`cpp-httplib`、`nlohmann/json`、`utf8proc` **不在上表中**，原因见第 4 节。

也可以直接用工作区脚本，它带安装后自检：

```bash
sudo tools/verify/install_deps_rocky10.sh install   # 安装并自检
tools/verify/install_deps_rocky10.sh check          # 只自检，不改系统
```

## 2. 依赖清单与来源对应

| 系统包 | 提供的能力 | CMake 中的消费点 | 本机是否已装 |
|---|---|---|---|
| `cmake` | 构建系统，要求 >= 3.28 | `cmake_minimum_required(VERSION 3.28)` | 否 |
| `ninja-build` | CMake 生成器 | 构建命令用 `-G Ninja` | 是（1.11.1-9.el10） |
| `pkgconf-pkg-config` | `pkg_check_modules` 后端 | `find_package(PkgConfig REQUIRED)` | 是（2.1.0-3.el10） |
| `libavformat-free-devel` | `libavformat/avformat.h` + `libavformat.pc` | `pkg_check_modules(FFMPEG REQUIRED ... libavformat>=60)` | 否 |
| `libavcodec-free-devel` | `libavcodec/avcodec.h` + `libavcodec.pc` | 同上，`libavcodec>=60` | 否 |
| `libavutil-free-devel` | `libavutil/{imgutils,pixdesc,display,error}.h` + `.pc` | 同上，`libavutil>=58` | 否 |
| `libswscale-free-devel` | `libswscale/swscale.h` + `.pc` | 同上，`libswscale>=7` | 否 |
| `libcurl-devel` | `curl/curl.h` + `libcurl.pc` | `pkg_check_modules(LIBCURL REQUIRED ... libcurl>=7.85)` | 否 |

版本下限全部来自源码，不是选定的：`CMakeLists.txt:127` 要求 `libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7`，`CMakeLists.txt:131` 要求 `libcurl>=7.85`。

## 3. 两个非显然的门控条件

### 3.1 Linux 下 FFmpeg 是强制的，关掉 apps 也躲不掉

`CMakeLists.txt` 的非 WIN32 分支里，FFmpeg 查找位于顶层，不在任何 `option` 之内：

```cmake
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  ...
```

因此 `-DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF` 仍然会在 configure 阶段因缺 FFmpeg 而失败。四件套必须装。

### 3.2 libcurl 只在构建 apps 或测试时需要

`NINFER_BUILD_MEDIA_ACQUIRE` 在 `NINFER_BUILD_APPS` 或 `BUILD_TESTING` 打开时被置为 ON（`CMakeLists.txt:93-96`），此时才追加 `pkg_check_modules(LIBCURL REQUIRED ...)`。

如果只想构建引擎本体：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF
```

这样可以少装 `libcurl-devel`；FFmpeg 四件套依然要装。

## 4. 看起来缺、实际不缺的头文件

对 `src/ apps/ include/ tests/` 下全部 110 个尖括号 include 逐个做 `g++ -fsyntax-only` 探测，共 39 个在本机搜不到。逐个核对后，只有 9 个是真缺（第 2 节表中那些），其余是探测器自身的假阳性或平台无关头：

| 类别 | 头文件 | 为什么不算缺 |
|---|---|---|
| 平台分支 | `windows.h` `winsock2.h` `ws2tcpip.h` `process.h` `io.h` `d3d12.h` `dxgi1_6.h` `dstorage.h` `dstorageerr.h` `wrl/client.h` | 全部在 `#ifdef _WIN32` 内，Linux 构建不会展开 |
| CMake 生成 | `ninfer/targets/qwen3_6/**`、`ninfer/targets/qwen3_6_27b/package.h` 等 | 来自 `src/targets/qwen3_6/export/`，由 CMake 配置生成 include 目录 |
| 内置源码树 | `nlohmann/json.hpp`、`utf8proc/utf8proc.h`、`httplib.h` | 已随仓内置于 `third_party/`，`src/CMakeLists.txt:5` 与 `:303` 已接线 |
| CUDA 13 布局 | `cub/block/block_merge_sort.cuh`、`cuda_pipeline.h` | CUDA 13 将 CUB 移到 `$CUDA/include/cccl/cub`，由 `CUDAToolkit` 的 include 目录提供；探测命令未带该路径 |
| 构建期拉取 | `xgrammar/compiler.h`、`matcher.h`、`tokenizer_info.h` | 见 4.1 |

### 4.1 xgrammar 不是系统包，装不了也不用装

xgrammar 由 `FetchContent_Declare` 在 configure 阶段从 GitHub 拉取（`CMakeLists.txt:167-193`，`GIT_TAG v0.2.5.post1`）。它不是 dnf 可解的依赖，本机 `dnf provides '*/xgrammar/compiler.h'` 返回无匹配。

要求：`git` 已装（本机 2.52.0），且 configure 阶段能访问 `github.com`。若构建机离线，需要预先准备 `FETCHCONTENT_SOURCE_DIR_XGRAMMAR_SOURCE` 指向本地副本。

`.gitmodules` 不存在，CUB/Thrust 等不通过子模块获取。

## 5. 本机实测校验

环境：`Rocky Linux 10.2 (Red Quartz)`，dnf 4.20.0，gcc/g++ 14.3.1，nvcc 13.3（CMakeLists 要求 >= 12.8）。

已启用仓库：`baseos` `appstream` `crb` `extras` `epel` `cuda-rhel10-x86_64`。`epel-release-10-8.el10_2` 已安装。

上面的包在本机全部可解，版本均满足源码下限：

| 包 | 候选版本 | 来源仓库 | 对照下限 |
|---|---|---|---|
| `cmake` | 3.31.8-1.el10 | appstream | >= 3.28 满足 |
| `libcurl-devel` | 8.12.1-4.el10_2.4 | appstream | >= 7.85 满足 |
| `libavformat-free-devel` | 7.1.2-1.el10_2（soname `libavformat.so.61`） | epel | >= 60 满足 |
| `libavcodec-free-devel` | 7.1.2-1.el10_2（soname `libavcodec.so.61`） | epel | >= 60 满足 |
| `libavutil-free-devel` | 7.1.2-1.el10_2（soname `libavutil.so.59`） | epel | >= 58 满足 |
| `libswscale-free-devel` | 7.1.2-1.el10_2（soname `libswscale.so.8`） | epel | >= 7 满足 |

事务规模（`dnf install --assumeno` 实测）：

- 第 1 节的精简组合：**98 个包，下载 52 MB，安装后 146 MB**（配合 `--setopt=install_weak_deps=False`）
- 若改用 `ffmpeg-free-devel` 元包：**241 个包，下载 114 MB**，因为它连带 `libavdevice`/`libavfilter`，进而拖入 pipewire、samba、tesseract 等桌面栈

所以命令里刻意用四个 `-free-devel` 子包，而不是 `ffmpeg-free-devel`。

## 6. 备选方案

### 6.1 需要完整编解码器时改用 RPM Fusion

EPEL 的 `ffmpeg-free` 是剔除专利受限组件的构建。若媒体采集阶段要用到 `ffmpeg-free` 未收录的编解码器，改用 RPM Fusion 的完整 `ffmpeg-devel`：

```bash
sudo dnf install -y \
  https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm
sudo dnf install -y --setopt=install_weak_deps=False ffmpeg-devel libcurl-devel cmake
```

注意 `ffmpeg-free` 与 RPM Fusion 的 `ffmpeg` 互斥，切换前需先卸载前者。本机未启用 RPM Fusion，`ffmpeg-free` 与现有包无冲突（`dnf repoquery --conflicts ffmpeg-free` 无输出）。

### 6.2 与上游 Dockerfile 的对照

`ninfer-4090/Dockerfile` 用 Ubuntu 24.04 基础镜像，安装 `cmake libavcodec-dev libavformat-dev libavutil-dev libcurl4-openssl-dev libswscale-dev ninja-build pkg-config`。本表逐项对应，无遗漏项。

## 7. 安装后自检与构建

```bash
tools/verify/install_deps_rocky10.sh check

export NINFER_ROOT=/root/ninfer-4090
tools/verify/build.sh incremental -- -DNINFER_BUILD_APPS=ON
```

自检覆盖：工具链可执行文件、5 个 pkg-config 模块及其版本下限、以及一个同时 include ffmpeg 与 curl 头文件的编译探针。

构建阶段除系统包外还依赖网络（xgrammar 拉取）与 WebUI 资源下载（`NINFER_ENABLE_UI` 默认 ON，可从 GitHub release 拉取；离线时 CMake 会告警并退化为 UI stub）。

### 7.1 用 uv tool install 安装时

`uv tool install` 走的正是这条构建链路 —— 它在 wheel 构建阶段现场克隆上游并编译，所以上面的
5 个系统包一个都不能少，此外还需要：

| 额外依赖 | 用途 | 备注 |
|---|---|---|
| `uv` | 安装器 | 官方脚本一行装好，见下 |
| CUDA Toolkit（含 `nvcc`） | 编译三元内核 | 需与本机驱动匹配 |
| `git` | 拉取上游与 xgrammar | 见 [把本仓当工具用](uv-工具安装.md) |

    curl -LsSf https://astral.sh/uv/install.sh | sh

网络不可达时，可以先在一台联网机器上 `uv build --wheel`，再把 wheel 拷到目标机安装；
也可以用 `NINFER_TERNARY_ENABLE_UI=0` 跳过 WebUI 资源下载。

## 8. 未验证项

- 第 5 节的事务规模来自 `dnf install --assumeno` 解析，未实际执行安装；包版本与仓库归属来自 `dnf list --available` 与 `dnf repoquery`。
- `ffmpeg-free` 相对 RPM Fusion 完整构建具体缺哪些解码器未逐个比对，仅记录为已知差异。
- 完整 CMake 构建（含 xgrammar 拉取与 UI 下载）尚未在本机跑通，依赖安装完成前无法执行。
