#!/usr/bin/env bash
# 构建 ninfer（Linux）。用法：build.sh [clean|incremental] [-- <额外的 cmake 参数>]
#
# 环境变量：
#   NINFER_ROOT         ninfer 源码树根目录（必填）
#   NINFER_BUILD_ROOT   构建目录（默认 <NINFER_ROOT>/build）
#   NINFER_ARCH         CUDA 架构，86 或 89（默认 89）
set -euo pipefail

mode="${1:-incremental}"
shift || true
# 文档中的 "-- <额外参数>" 是可选的视觉分隔符；不剥掉会被 cmake 当成未知参数。
if [[ "${1:-}" == "--" ]]; then
  shift
fi

: "${NINFER_ROOT:?请先设置 NINFER_ROOT=<ninfer 源码树根目录>}"
arch="${NINFER_ARCH:-89}"
build_root="${NINFER_BUILD_ROOT:-${NINFER_ROOT}/build}"
log="${build_root}/${mode}.log"

if [[ "${arch}" != "86" && "${arch}" != "89" ]]; then
  echo "NINFER_ARCH 只能是 86 或 89，实际为 ${arch}" >&2
  exit 2
fi

cmake -S "${NINFER_ROOT}" -B "${build_root}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
  "${@}"

mkdir -p "${build_root}"
echo "=== build ${mode} start $(date -Is) ===" > "${log}"
if [[ "${mode}" == "clean" ]]; then
  cmake --build "${build_root}" --target clean >> "${log}" 2>&1 || true
fi

set +e
cmake --build "${build_root}" -j "${NINFER_JOBS:-8}" >> "${log}" 2>&1
rc=$?
set -e

echo "=== exit code: ${rc} ===" >> "${log}"
echo "MODE=${mode} EXIT=${rc} LOG=${log}"
exit "${rc}"
