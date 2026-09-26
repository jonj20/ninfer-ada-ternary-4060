#!/usr/bin/env bash
# 各 (KV 量化, max-context) 组合的可用性探针：服务能否起来、各池占用、剩余显存。
#
# 8 GB 卡上三元 27B 的权重就占掉大半，KV 池是剩下来的那一点，所以「能开多大上下文」完全由
# kv-dtype 决定，而这个上限只能实测（布局里有 MTP / vision 的额外页，静态算不出来）。
#
# 用法：kv_capacity_probe.sh <artifact.ninfer> [--pairs "rk4v4-e8:49152,rk4v4:32768,..."]
#
# 环境变量：
#   NINFER_BUILD_ROOT  构建目录（默认 build）；serve 取 <root>/apps/ninfer-serve
#   NINFER_PORT        端口（默认 8080）
#   NINFER_PROBE_LOG   日志路径（默认 mktemp，退出时删除）
set -euo pipefail

artifact="${1:?用法：kv_capacity_probe.sh <artifact.ninfer> [--pairs kv:ctx,...]}"
shift || true

# 默认探这组：e8 变体的上限最高，其次无 e8 的，最后两个 32768 作为「肯定能起来」的对照。
pairs="rk4v4-e8:49152,rk4v4-e8:65536,rk4v4:32768,rk4v4:65536,rk8v4:32768,rk8v4:65536,rk2v4-e8:65536"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pairs) pairs="$2"; shift 2 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

serve="${NINFER_BUILD_ROOT:-build}/apps/ninfer-serve"
port="${NINFER_PORT:-8080}"
log="${NINFER_PROBE_LOG:-$(mktemp -t ninfer-kvcap-XXXXXX.log)}"

if [[ ! -x "${serve}" ]]; then
  echo "缺少 serve 可执行文件: ${serve}（用 NINFER_BUILD_ROOT 指定构建目录）" >&2
  exit 99
fi
if [[ ! -f "${artifact}" ]]; then
  echo "缺少制品: ${artifact}" >&2
  exit 98
fi

cleanup() { pkill -f 'apps/ninfer-serve' 2>/dev/null || true; }
trap 'cleanup; [[ -z "${NINFER_PROBE_LOG:-}" ]] && rm -f "${log}" || true' EXIT

probe() {
  local kv="$1" ctx="$2"
  pkill -f 'apps/ninfer-serve' 2>/dev/null || true
  sleep 4
  rm -f "${log}"
  nohup "${serve}" "${artifact}" --host 127.0.0.1 --port "${port}" \
    --max-context "${ctx}" --kv-dtype "${kv}" --max-concurrency 1 \
    --default-max-tokens 2048 --no-thinking > "${log}" 2>&1 &
  local ok=0
  for _ in $(seq 1 32); do
    sleep 5
    if grep -q 'listening on' "${log}" 2>/dev/null; then ok=1; break; fi
    if grep -qiE '\[error\]|failed|cannot|not enough|exceeds|invalid' "${log}" 2>/dev/null; then break; fi
  done
  printf '%-12s ctx=%-7s ' "${kv}" "${ctx}"
  if [[ "${ok}" == "1" ]]; then
    printf 'OK   '
    grep -a 'KV capacity' "${log}" | sed 's/.*resolved=/resolved=/' | cut -c1-160 || true
    echo
  else
    printf 'FAIL '
    grep -aiE '\[error\]|failed|cannot|not enough|exceeds|invalid' "${log}" | head -1 | cut -c1-160 || true
    echo
  fi
}

echo "== 制品: $(basename "${artifact}") =="
IFS=',' read -r -a pair_list <<< "${pairs}"
for pair in "${pair_list[@]}"; do
  probe "${pair%%:*}" "${pair##*:}"
done
cleanup
echo "（已全部停止）"
