#!/usr/bin/env bash
# 三元制品端到端一致性验证：内核路径（MMA/SIMT）、prefill 分块、MTP 投机，外加一个负控。
#
# 判据只有一条 —— 同一 prompt 下的贪心 token 序列必须逐字节一致；负控必须不一致，
# 否则说明这个比对没有判别力。序列取自 CLI 的 --print-token-ids，不做文本近似比较。
#
# 用法：e2e_ternary.sh <artifact.ninfer> [prompt]
#
# 环境变量：
#   NINFER_CLI          ninfer 可执行文件（默认 <NINFER_BUILD_ROOT>/apps/ninfer）
#   NINFER_E2E_OUT      输出目录（默认 ./out/e2e-<制品名>）
#                       结果是证据，脚本既不写进制品目录，也不自动删除
#   NINFER_E2E_MAX_NEW  生成长度（默认 192）
set -euo pipefail

artifact="${1:?用法：e2e_ternary.sh <artifact.ninfer> [prompt]}"
prompt="${2:-A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How much does the ball cost? Reason briefly, then state the answer.}"
cli="${NINFER_CLI:-${NINFER_BUILD_ROOT:-build}/apps/ninfer}"
max_new="${NINFER_E2E_MAX_NEW:-192}"
out_dir="${NINFER_E2E_OUT:-out/e2e-$(basename "${artifact}" .ninfer)}"

if [[ ! -x "${cli}" ]]; then
  echo "缺少 CLI 可执行文件: ${cli}" >&2
  exit 99
fi
if [[ ! -f "${artifact}" ]]; then
  echo "缺少制品: ${artifact}" >&2
  exit 98
fi
mkdir -p "${out_dir}"
index="${out_dir}/index.tsv"
: > "${index}"

# 每条用例跑一次生成，把 token 序列落进 index.tsv。envspec 与 extra 故意不加引号展开：
# 它们是被用作"环境变量前缀 + 附加参数"的片段，不是单个参数。
run_case() {
  local label="$1" envspec="$2" extra="$3" rc=0 ids=""
  # shellcheck disable=SC2086
  env ${envspec} timeout 900 "${cli}" "${artifact}" \
    --prompt "${prompt}" --max-new "${max_new}" --max-context 2048 \
    --no-thinking --greedy --seed 1234 --print-token-ids ${extra} \
    > "${out_dir}/${label}.out" 2> "${out_dir}/${label}.err" || rc=$?
  ids="$(grep -m1 "generated ids" "${out_dir}/${label}.err" | sed "s/.*generated ids *//" || true)"
  # 摘要取 id 列表加一个换行，与 `echo "$ids" | md5sum` 的口径一致，便于和手工复核对照。
  printf "%s\t%s\t%s\n" "${label}" "${rc}" "$(printf '%s\n' "${ids}" | md5sum | cut -c1-12)" >> "${index}"
}

echo "== 产物: $(basename "${artifact}") =="
run_case mma1-c128   "NINFER_TERNARY_MMA=1" "--prefill-chunk 128"
run_case mma1-c1024  "NINFER_TERNARY_MMA=1" "--prefill-chunk 1024"
run_case mma0-c128   "NINFER_TERNARY_MMA=0" "--prefill-chunk 128"
run_case mma0-c1024  "NINFER_TERNARY_MMA=0" "--prefill-chunk 1024"
run_case mtp4        "" "--spec mtp --draft-tokens 4 --lm-head-draft"
run_case neg-norot   "NINFER_TERNARY_HADAMARD=0" "--prefill-chunk 1024"

cat "${index}"

base="$(awk -F"\t" '$1=="mma1-c1024"{print $3}' "${index}")"
fail=0
while IFS=$'\t' read -r label rc hash; do
  if [[ "${rc}" != "0" ]]; then
    echo "FAIL ${label}: 退出码 ${rc}，见 ${out_dir}/${label}.err"
    fail=1
    continue
  fi
  if [[ -z "${hash}" || "${hash}" == "d41d8cd98f00" ]]; then
    echo "FAIL ${label}: 没有取到 token 序列"
    fail=1
  fi
done < "${index}"

# 正控：除负控外必须与基准逐字节一致。
while IFS=$'\t' read -r label rc hash; do
  [[ "${label}" == "neg-norot" ]] && continue
  if [[ "${hash}" != "${base}" ]]; then
    echo "FAIL ${label}: token 序列与基准不一致 (${hash} != ${base})"
    fail=1
  fi
done < "${index}"

# 负控：关掉折叠基旋转必须改变输出，否则比对没有判别力。
neg="$(awk -F"\t" '$1=="neg-norot"{print $3}' "${index}")"
if [[ "${neg}" == "${base}" ]]; then
  echo "FAIL neg-norot: 关掉旋转后输出没变，比对无判别力"
  fail=1
fi

echo "--- MTP 接受率 ---"
grep -h "mtp acceptance" "${out_dir}/mtp4.err" || echo "  (无)"

if [[ "${fail}" == "0" ]]; then
  echo "RESULT: PASS（基准 ${base}；负控 ${neg}）"
else
  echo "RESULT: FAIL"
fi
exit "${fail}"