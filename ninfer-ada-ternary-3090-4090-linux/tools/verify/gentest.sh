#!/usr/bin/env bash
# 可复现的生成探针。用法：gentest.sh <label> <artifact> <prompt> <max-new> [额外参数...]
#
# 环境变量：
#   NINFER_CLI   ninfer CLI 可执行文件（默认 <NINFER_BUILD_ROOT>/apps/ninfer）
#   NINFER_LOG_DIR  日志目录（默认当前目录）
set -euo pipefail

label="${1:?用法：gentest.sh <label> <artifact> <prompt> <max-new> [额外参数...]}"
artifact="${2:?}"
prompt="${3:?}"
max_new="${4:?}"
shift 4

cli="${NINFER_CLI:-${NINFER_BUILD_ROOT:-build}/apps/ninfer}"
log_dir="${NINFER_LOG_DIR:-.}"
log="${log_dir}/gen-${label}.log"

if [[ ! -x "${cli}" ]]; then
  echo "缺少 CLI 可执行文件: ${cli}" >&2
  exit 99
fi
if [[ ! -f "${artifact}" ]]; then
  echo "缺少制品: ${artifact}" >&2
  exit 98
fi

mkdir -p "${log_dir}"
echo "=== gen test ${label} start $(date -Is) ===" > "${log}"

set +e
"${cli}" "${artifact}" --prompt "${prompt}" --max-new "${max_new}" "${@}" >> "${log}" 2>&1
rc=$?
set -e

echo "=== exit code: ${rc} ===" >> "${log}"
echo "LABEL=${label} EXIT=${rc} LOG=${log}"
exit "${rc}"
