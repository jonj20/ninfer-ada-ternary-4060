#!/usr/bin/env bash
# 一键编译 RTX 4060 (sm_89) → build_4060/apps/ninfer
# 用法: ./build_4060.sh          # 增量
#       ./build_4060.sh clean    # 从零
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode="${1:-incremental}"
echo "==> build sm_89 mode=${mode} root=$(pwd)"
exec tools/verify/build.sh "${mode}" -- -DNINFER_BUILD_APPS=ON
