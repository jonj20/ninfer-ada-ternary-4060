#!/usr/bin/env bash
# 一键编译 RTX 4060 (sm_89) → build_4060/apps/ninfer
# 用法: ./build_4060.sh          # 增量
#       ./build_4060.sh clean    # 从零
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 固定产物目录：build_4060/（build.sh 未设时默认是 <root>/build）
export NINFER_BUILD_ROOT="${NINFER_BUILD_ROOT:-$PWD/build_4060}"

# 7.6 GiB 内存档（4060 便携机默认）：单文件极限编译需 -j1，否则 CUDA 编译 OOM。
# 显式设 NINFER_JOBS 可覆盖。
export NINFER_JOBS="${NINFER_JOBS:-1}"

mode="${1:-incremental}"
echo "==> build sm_89 mode=${mode} root=$(pwd) build_dir=${NINFER_BUILD_ROOT} jobs=${NINFER_JOBS}"
exec tools/verify/build.sh "${mode}" -- -DNINFER_BUILD_APPS=ON
