#!/usr/bin/env bash
# 可复现的制品装载探针。用法：loadtest.sh <label> <artifact.ninfer>
#
# 环境变量：
#   NINFER_CLI   ninfer CLI 可执行文件（默认 <NINFER_BUILD_ROOT>/apps/ninfer）
#   NINFER_LOG_DIR  日志目录（默认当前目录）
set -euo pipefail

label="${1:?用法：loadtest.sh <label> <artifact.ninfer>}"
artifact="${2:?用法：loadtest.sh <label> <artifact.ninfer>}"

cli="${NINFER_CLI:-${NINFER_BUILD_ROOT:-build}/apps/ninfer}"
log_dir="${NINFER_LOG_DIR:-.}"
log="${log_dir}/load-${label}.log"

if [[ ! -x "${cli}" ]]; then
  echo "缺少 CLI 可执行文件: ${cli}" >&2
  exit 99
fi
if [[ ! -f "${artifact}" ]]; then
  echo "缺少制品: ${artifact}" >&2
  exit 98
fi

mkdir -p "${log_dir}"
{
  echo "=== load test ${label} start $(date -Is) ==="
  echo "artifact: ${artifact}"
  echo "exe: ${cli}"
} > "${log}"

set +e
"${cli}" "${artifact}" --prompt hi --max-new 1 >> "${log}" 2>&1
rc=$?
set -e

echo "=== exit code: ${rc} ===" >> "${log}"
echo "LABEL=${label} EXIT=${rc} LOG=${log}"
exit "${rc}"
