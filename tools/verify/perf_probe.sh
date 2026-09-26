#!/usr/bin/env bash
# 三元制品的吞吐探针：解码、长 prompt 预填充、MTP 投机三个口径一次跑完。
#
# 和 e2e_ternary.sh 的分工：那个管**正确性**（贪心 token 序列逐字节比对 + 负控），这个管**性能**
# （tok/s，以及 MTP 的 tok/round 与每轮成本）。改解码内核、改 tile 分派、改投机路径之后跑这个。
#
# 用法：perf_probe.sh <artifact.ninfer> [--drafts 0,1,2,3] [--kv rk4v4-e8] [--ctx 8192]
#
# 环境变量：
#   NINFER_BUILD_ROOT  构建目录（默认 build）；serve 可执行文件取 <root>/apps/ninfer-serve
#   NINFER_PORT        端口（默认 8080）
#   NINFER_PROMPT      自定义 prompt（默认一段中文技术说明，约 25 token）
#   NINFER_LONG_PROMPT 长 prompt 用的重复单元（默认内置中文技术段落，约 4692 token）
#   NINFER_MAX_TOKENS  单请求生成长度（默认 512）
set -euo pipefail

artifact="${1:?用法：perf_probe.sh <artifact.ninfer> [--drafts ...] [--kv ...] [--ctx ...]}"
shift || true

drafts="0,1,2,3"
kv="rk4v4-e8"
ctx="8192"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --drafts) drafts="$2"; shift 2 ;;
    --kv)     kv="$2";     shift 2 ;;
    --ctx)    ctx="$2";    shift 2 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

serve="${NINFER_BUILD_ROOT:-build}/apps/ninfer-serve"
port="${NINFER_PORT:-8080}"
max_tokens="${NINFER_MAX_TOKENS:-512}"
prompt="${NINFER_PROMPT:-请用 300 字说明什么是张量并行。}"
long_prompt="${NINFER_LONG_PROMPT:-张量并行把权重矩阵按行或列切到多张卡上，每张卡只算一部分输出，再靠 AllReduce 或 AllGather 把结果拼回来。流水线并行把层切开，让不同批次错开峰值。}"

if [[ ! -x "${serve}" ]]; then
  echo "缺少 serve 可执行文件: ${serve}（用 NINFER_BUILD_ROOT 指定构建目录）" >&2
  exit 99
fi
if [[ ! -f "${artifact}" ]]; then
  echo "缺少制品: ${artifact}" >&2
  exit 98
fi

log="$(mktemp -t ninfer-perf-XXXXXX.log)"
out="$(mktemp -d -t ninfer-perf-XXXXXX)"
trap 'pkill -f "apps/ninfer-serve" 2>/dev/null || true; rm -rf "${log}" "${out}"' EXIT

# 启动一次服务，等就绪。MTP 的 draft window 是启动参数，所以每个档位都要重启。
start_serve() {
  local spec="$1"
  pkill -f 'apps/ninfer-serve' 2>/dev/null || true
  sleep 3
  rm -f "${log}"
  # shellcheck disable=SC2086
  nohup "${serve}" "${artifact}" --host 127.0.0.1 --port "${port}" \
    --max-context "${ctx}" --kv-dtype "${kv}" --max-concurrency 1 \
    --default-max-tokens "${max_tokens}" --no-thinking ${spec} > "${log}" 2>&1 &
  for _ in $(seq 1 40); do
    sleep 5
    grep -q 'listening on' "${log}" 2>/dev/null && return 0
    grep -qiE '\[error\]|failed|cannot|not enough' "${log}" 2>/dev/null && return 1
  done
  return 1
}

# 一次请求。draft=0 表示不开投机。输出 "decode_tok_s gen prefill_tok_s prompt_tokens" 到 stdout。
request() {
  local max_out="$1" body="$2"
  python3 - "${port}" "${max_out}" "${body}" <<'PY'
import json, sys, urllib.request
port, max_out, body = sys.argv[1], int(sys.argv[2]), sys.argv[3]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"model": "local", "temperature": 0, "max_tokens": max_out,
                     "messages": [{"role": "user", "content": body}]}).encode(),
    headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
t = d["timings"]
print(f"{t['predicted_per_second']:.2f} {d['usage']['completion_tokens']} "
      f"{t['prompt_per_second']:.1f} {d['usage']['prompt_tokens']}")
PY
}

echo "== 制品: $(basename "${artifact}")  kv=${kv} ctx=${ctx} =="

# ---- 1) 解码基线 + MTP 扫描 ----------------------------------------------------
# MTP 的判据不是 tok/s 本身，而是「每轮成本」：decode 秒数 / 轮数，轮数 = gen / tok_per_round。
# tok_per_round 与接受率从 serve 的请求日志里取，格式见 generation_service 的 speculative 统计。
printf '%-10s %10s %8s %12s %10s %s\n' 档位 decode_tok_s gen tok/round 每轮ms 备注
IFS=',' read -r -a draft_list <<< "${drafts}"
for d in "${draft_list[@]}"; do
  spec=""
  [[ "${d}" != "0" ]] && spec="--spec mtp --draft-tokens ${d}"
  if ! start_serve "${spec}"; then
    printf '%-10s %s\n' "draft=${d}" "启动失败，见 ${log}"
    tail -3 "${log}" || true
    continue
  fi
  read -r dec gen pre ptok < <(request "${max_tokens}" "${prompt}")
  line="$(grep -a 'done finish' "${log}" | tail -1 || true)"
  spec_note="$(printf '%s' "${line}" | sed -n 's/.*\(speculative=[a-z]*\( [0-9.]*tok\/round ([0-9.]*%)\)\?\).*/\1/p' || true)"
  if [[ -z "${spec_note}" ]]; then spec_note="speculative=off"; fi
  tpr="$(printf '%s' "${spec_note}" | sed -n 's/.* \([0-9.]*\)tok\/round.*/\1/p' || true)"
  # 每轮成本 = 轮数换算：decode 秒数 = 轮数 × 每轮秒数，轮数 = gen / tok_per_round，
  # 于是每轮秒数 = (gen / decode_tok_s) / (gen / tok_per_round) = tok_per_round / decode_tok_s。
  per_round="-"
  if [[ -n "${tpr}" && "${dec}" != "0" && "${dec}" != "-" ]]; then
    per_round="$(awk -v t="${tpr}" -v d="${dec}" 'BEGIN{printf "%.1f", 1000*t/d}')"
  fi
  printf '%-10s %10s %8s %12s %10s %s\n' "draft=${d}" "${dec}" "${gen}" \
    "${tpr:--}" "${per_round:--}" "${spec_note}"
done

# ---- 2) 长 prompt 预填充回归 --------------------------------------------------
# 短 prompt 的 prefill 速率被每请求固定开销主导（实测 13 token 的 prefill 只有 20~55 tok/s，
# 600 token 约 110 tok/s），拿它判回归会误判，所以这里用长 prompt，并且必须连 token 数一起看：
# 同一个内核在 1572 / 3132 / 4692 token 上分别是 129 / 132 / 131 tok/s，600 token 只有 110。
# 另一个坑：连续两次发**内容完全相同**的 prompt 会命中前缀缓存，第二次的 prefill 几乎是 0，
# 拿它当重复性测量会得出 250% 的「波动」。
if start_serve ""; then
  long_body="${long_prompt}"
  for _ in $(seq 1 39); do long_body="${long_body}${long_prompt}"; done
  read -r dec gen pre ptok < <(request 64 "${long_body}")
  printf '\n长 prompt prefill: %s tok/s（%s token；生成 %s token 时 decode %s tok/s）\n' \
    "${pre}" "${ptok}" "${gen}" "${dec}"
  printf '参考：3000 token 以上时本卡约 130 tok/s。token 数偏小则数值偏低，不要横向比较。\n'
else
  printf '\n长 prompt prefill: 启动失败\n'
fi

pkill -f 'apps/ninfer-serve' 2>/dev/null || true
echo
echo "（已停止）"
