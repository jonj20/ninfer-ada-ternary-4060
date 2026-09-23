#!/usr/bin/env bash
# 构建 ninfer（Linux）。用法：build.sh [clean|incremental] [-- <额外的 cmake 参数>]
#
# 环境变量：
#   NINFER_ROOT         ninfer 源码树根目录；默认=本仓根（源码已合入，无需外置检出）
#   NINFER_BUILD_ROOT   构建目录（默认 <NINFER_ROOT>/build）
#   NINFER_ARCH         CUDA 架构，86 或 89（默认 89）
set -euo pipefail

mode="${1:-incremental}"
shift || true
# 文档中的 "-- <额外参数>" 是可选的视觉分隔符；不剥掉会被 cmake 当成未知参数。
if [[ "${1:-}" == "--" ]]; then
  shift
fi

# 脚本位于 <repo>/tools/verify/build.sh → 仓根为三级父目录。
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd "${_here}/../.." && pwd)"
NINFER_ROOT="${NINFER_ROOT:-${_repo_root}}"
export NINFER_ROOT

if [[ ! -f "${NINFER_ROOT}/CMakeLists.txt" || ! -d "${NINFER_ROOT}/src" ]]; then
  echo "NINFER_ROOT=${NINFER_ROOT} 不是含 CMakeLists.txt 与 src/ 的源码树" >&2
  exit 2
fi

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
echo "NINFER_ROOT=${NINFER_ROOT} BUILD_ROOT=${build_root} ARCH=${arch}"
exit "${rc}"
