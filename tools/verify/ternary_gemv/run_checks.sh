#!/usr/bin/env bash
# 编译并运行 PTQ1_0 解码 GEMV 的检查：数值对拍、带宽、访存归因、int8 精度。
#
# 用法：
#   tools/verify/ternary_gemv/run_checks.sh              # 全部
#   tools/verify/ternary_gemv/run_checks.sh reference    # 只跑数值对拍（最快，改完内核先跑这个）
#   tools/verify/ternary_gemv/run_checks.sh bandwidth    # 只跑带宽
#   tools/verify/ternary_gemv/run_checks.sh pattern      # 只跑访存归因
#   tools/verify/ternary_gemv/run_checks.sh accuracy     # 只跑 int8 精度
#
# 环境变量：
#   NINFER_ROOT   源码树根（含 src/ 与 CMakeLists.txt）；默认=本仓根
#   NINFER_ARCH   CUDA 架构，默认 89（4060/4090）
#   NVCC          nvcc 路径；默认取 PATH 上的
#   WORK_DIR      产物目录，默认 <repo>/out/ternary-gemv
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo_root="$(cd "${_here}/../../.." && pwd)"
NINFER_ROOT="${NINFER_ROOT:-${_repo_root}}"
export NINFER_ROOT

if [[ ! -d "${NINFER_ROOT}/src/ops/linear/ternary" ]]; then
  echo "NINFER_ROOT=${NINFER_ROOT} 下没有 src/ops/linear/ternary" >&2
  exit 2
fi

arch="${NINFER_ARCH:-89}"
NVCC="${NVCC:-nvcc}"
WORK_DIR="${WORK_DIR:-${_repo_root}/out/ternary-gemv}"
mkdir -p "${WORK_DIR}"

which="${1:-all}"
rc=0

# 需要读引擎头文件（要 -I src）的检查
build_with_engine() {
  local name="$1"
  echo "=== build ${name} (sm_${arch}) ==="
  "${NVCC}" -O3 -std=c++17 "-arch=sm_${arch}" -I"${NINFER_ROOT}/src" \
    "${_here}/${name}.cu" -o "${WORK_DIR}/${name}"
}

# 独立探针（不读引擎头文件）
build_standalone() {
  local name="$1"
  echo "=== build ${name} (sm_${arch}) ==="
  "${NVCC}" -O3 -std=c++17 "-arch=sm_${arch}" \
    "${_here}/${name}.cu" -o "${WORK_DIR}/${name}"
}

case "${which}" in
  reference) build_with_engine gemv_reference_check; build_with_engine prefill_reference_check ;;
  bandwidth) build_with_engine gemv_bandwidth ;;
  accuracy)  build_with_engine activation_quant_accuracy ;;
  pattern)   build_standalone pattern_attribution ;;
  all)
    build_with_engine gemv_reference_check
    build_with_engine prefill_reference_check
    build_with_engine gemv_bandwidth
    build_with_engine activation_quant_accuracy
    build_standalone pattern_attribution
    ;;
  *) echo "未知参数：${which}（可用：reference|bandwidth|accuracy|pattern|all）" >&2; exit 2 ;;
esac

if [[ "${which}" == "reference" || "${which}" == "all" ]]; then
  echo
  echo "=== 数值对拍：解码内核 vs CPU 参考（容差 = bf16 舍入）==="
  if "${WORK_DIR}/gemv_reference_check"; then
    echo "reference: PASS"
  else
    echo "reference: FAIL" >&2
    rc=1
  fi
  echo
  echo "=== 数值对拍：批量 prefill 内核 vs CPU 参考 ==="
  if "${WORK_DIR}/prefill_reference_check"; then
    echo "prefill_reference: PASS"
  else
    echo "prefill_reference: FAIL" >&2
    rc=1
  fi
fi

if [[ "${which}" == "bandwidth" || "${which}" == "all" ]]; then
  echo
  echo "=== 带宽（对照 tools/hbm_bandwidth_probe.cu 的可达上限）==="
  "${WORK_DIR}/gemv_bandwidth"
fi

if [[ "${which}" == "pattern" || "${which}" == "all" ]]; then
  echo
  echo "=== 访存归因（默认 312 MB 冷载荷；每加一层工作看掉多少）==="
  "${WORK_DIR}/pattern_attribution"
fi

if [[ "${which}" == "accuracy" || "${which}" == "all" ]]; then
  echo
  echo "=== int8 激活精度（对照 bf16 路径）==="
  "${WORK_DIR}/activation_quant_accuracy"
fi

echo
echo "WORK_DIR=${WORK_DIR}"
exit "${rc}"
