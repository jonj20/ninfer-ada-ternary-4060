#!/usr/bin/env bash
# ninfer_bench 的标准化封装：钉死语料、重复次数、预热与 prefill 分块，结果落成一张 tidy CSV，
# 并把可比性所依赖的环境一并记进 manifest.txt。
#
# 存在的理由：ninfer_bench 的默认语料是相对 CWD 的路径，重复/预热/分块又各有默认值，手敲一次
# 很容易把这几样一起漏掉 —— 那样两次跑出来的数字不可比，而"不可比"肉眼看不出来。本脚本把它们
# 钉死，并让每条用例都落到同一个 CSV 里，不同时间、不同制品之间可以直接对列。
#
# 用法：bench.sh <artifact.ninfer> [suite ...]
#
# suite（缺省 standard）：
#   standard  pp512 / pp2048 / tg128 / pp512+tg128 —— 面板基线，一次装载跑完
#   prefill   pp512 / pp2048 / pp8192 —— 长 prefill 扩展性
#   decode    tg128 / tg512 —— 稳态解码
#   kv        bf16 / int8 / rk8v4 / rk4v4 / rk4v4-e8 / rk2v4-e8 各一条 pp512+tg128
#   mtp       投机窗口 0 / 4 / 8；窗口 8 显式关 CUDA Graph，规避与本移植无关的上游缺陷
#   graph     CUDA Graph 开 / 关
#   all       以上全部
#
# 环境变量：
#   NINFER_BENCH         ninfer_bench 可执行文件（默认在若干常见构建目录里找）
#   NINFER_ROOT          ninfer 源码树，用来定位默认语料与记录引擎修订
#   NINFER_BUILD_ROOT    构建目录
#   NINFER_BENCH_OUT     输出目录（默认 ./out/bench-<制品名>-<UTC 时间戳>）
#                        结果是证据，脚本既不写进制品目录，也不自动删除
#   NINFER_BENCH_CORPUS  语料文件（默认 <NINFER_ROOT>/bench/fixtures/bench_corpus.ids）
#   NINFER_BENCH_REPS    重复次数（默认 5）
#   NINFER_BENCH_WARMUP  预热次数（默认 1）
#   NINFER_BENCH_CHUNK   prefill 分块（默认 1024，必须是 128 的倍数）
#   NINFER_BENCH_DEVICE  CUDA 设备序号（默认不传）
#   NINFER_BENCH_EXTRA   追加到每条用例的额外参数（按词分割）
#   NINFER_BENCH_NO_HASH=1  跳过制品摘要（制品很大时省一次全文件读取）
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  # 打印文件头部那段注释，去掉前导 "# "，不写死行号（改了注释就会漂）。
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  exit 0
fi

artifact="${1:?用法：bench.sh <artifact.ninfer> [suite ...]}"
shift
suites=("$@")
if [[ ${#suites[@]} -eq 0 ]]; then suites=(standard); fi

reps="${NINFER_BENCH_REPS:-5}"
warmup="${NINFER_BENCH_WARMUP:-1}"
chunk="${NINFER_BENCH_CHUNK:-1024}"
device="${NINFER_BENCH_DEVICE:-}"
extra="${NINFER_BENCH_EXTRA:-}"

if [[ ! -f "${artifact}" ]]; then
  echo "缺少制品: ${artifact}" >&2
  exit 98
fi
artifact="$(readlink -f "${artifact}")"

# 默认语料是相对 CWD 的路径：换一个工作目录就等于换一份语料，而输出里不会留下痕迹。
corpus="${NINFER_BENCH_CORPUS:-${NINFER_ROOT:-}/bench/fixtures/bench_corpus.ids}"
if [[ ! -f "${corpus}" ]]; then
  echo "缺少语料: ${corpus}" >&2
  echo "  用 NINFER_BENCH_CORPUS 指定，或先设置 NINFER_ROOT=<ninfer 源码树>" >&2
  exit 97
fi
corpus="$(readlink -f "${corpus}")"

resolve_bench() {
  if [[ -n "${NINFER_BENCH:-}" ]]; then
    printf "%s\n" "${NINFER_BENCH}"
    return 0
  fi
  local candidate
  for candidate in \
    "${NINFER_BUILD_ROOT:-}/bench/ninfer_bench" \
    "${NINFER_ROOT:-}/build/bench/ninfer_bench" \
    /data/ninfer-build-bench/bench/ninfer_bench \
    /data/ninfer-build/bench/ninfer_bench; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! bench="$(resolve_bench)"; then
  echo "找不到 ninfer_bench；用 NINFER_BENCH=<可执行文件> 指定" >&2
  exit 99
fi

# 每个 suite 展开成若干次 ninfer_bench 调用：一次调用装载一次制品，所以同参数能合并的用例
# 必须合并 —— 一个 20 GB 制品的装载时间与跑分本身同量级。
suite_cases() {
  case "$1" in
  standard)
    printf "%s\t%s\n" "panel" "-p 512,2048 -n 128 -pg 512,128"
    ;;
  prefill)
    printf "%s\t%s\n" "prefill" "-p 512,2048,8192"
    ;;
  decode)
    printf "%s\t%s\n" "decode" "-n 128,512"
    ;;
  kv)
    local kv
    for kv in bf16 int8 rk8v4 rk4v4 rk4v4-e8 rk2v4-e8; do
      printf "%s\t%s\n" "kv-${kv}" "-pg 512,128 --kv-dtype ${kv}"
    done
    ;;
  mtp)
    printf "%s\t%s\n" "mtp-off" "-pg 512,128"
    printf "%s\t%s\n" "mtp-draft4" "-pg 512,128 --mtp-draft-tokens 4 --lm-head-draft"
    # 上游缺陷：草稿窗口 >= 8 时 CUDA Graph 的 exec update 会失败。官方非三元制品同样复现，
    # 与本移植无关；这里显式关图，让这条用例测的是投机而不是那个缺陷。
    printf "%s\t%s\n" "mtp-draft8-nograph" \
      "-pg 512,128 --mtp-draft-tokens 8 --lm-head-draft --no-cuda-graph"
    ;;
  graph)
    printf "%s\t%s\n" "graph-on" "-pg 512,128"
    printf "%s\t%s\n" "graph-off" "-pg 512,128 --no-cuda-graph"
    ;;
  all)
    local nested
    for nested in standard prefill decode kv mtp graph; do suite_cases "${nested}"; done
    ;;
  *)
    echo "未知 suite: $1（可用：standard prefill decode kv mtp graph all）" >&2
    exit 2
    ;;
  esac
}

out_dir="${NINFER_BENCH_OUT:-out/bench-$(basename "${artifact}" .ninfer)-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${out_dir}"

sha_of() {
  if [[ "${NINFER_BENCH_NO_HASH:-0}" == "1" ]]; then
    printf "skipped\n"
  else
    sha256sum "$1" | cut -d" " -f1
  fi
}

{
  printf "artifact        %s\n" "${artifact}"
  printf "artifact_bytes  %s\n" "$(stat -c %s "${artifact}")"
  printf "artifact_sha256 %s\n" "$(sha_of "${artifact}")"
  printf "bench           %s\n" "${bench}"
  printf "bench_sha256    %s\n" "$(sha_of "${bench}")"
  printf "corpus          %s\n" "${corpus}"
  printf "corpus_sha256   %s\n" "$(sha_of "${corpus}")"
  printf "reps            %s\n" "${reps}"
  printf "warmup          %s\n" "${warmup}"
  printf "prefill_chunk   %s\n" "${chunk}"
  printf "device          %s\n" "${device:--}"
  printf "extra           %s\n" "${extra:--}"
  printf "suites          %s\n" "${suites[*]}"
  printf "started_utc     %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "gpu             %s\n" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  printf "driver          %s\n" "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  printf "engine_rev      %s\n" "$(git -C "${NINFER_ROOT:-.}" rev-parse HEAD 2>/dev/null || echo -)"
  printf "engine_dirty    %s\n" "$(git -C "${NINFER_ROOT:-.}" status --porcelain 2>/dev/null | wc -l)"
  printf -- "--- NINFER_* ---\n"
  env | grep -E "^NINFER_" | sort || true
} > "${out_dir}/manifest.txt"

combined="${out_dir}/results.csv"
header_done=0
cases=0
for suite in "${suites[@]}"; do
  while IFS=$'\t' read -r label args; do
    if [[ -z "${label}" ]]; then continue; fi
    case_label="${suite}__${label}"
    per_case="${out_dir}/${case_label}.csv"
    cmd=(--weights "${artifact}" --corpus "${corpus}"
         -r "${reps}" --warmup "${warmup}" --prefill-chunk "${chunk}"
         -o csv --output-file "${per_case}")
    # args 是"参数片段"，有意做词分割
    # shellcheck disable=SC2206
    cmd+=(${args})
    if [[ -n "${device}" ]]; then cmd+=(--device "${device}"); fi
    # shellcheck disable=SC2206
    if [[ -n "${extra}" ]]; then cmd+=(${extra}); fi
    printf "== %s: ninfer_bench %s\n" "${case_label}" "${cmd[*]}"
    "${bench}" "${cmd[@]}"
    if [[ ${header_done} -eq 0 ]]; then
      printf "case,suite,%s\n" "$(head -1 "${per_case}")" > "${combined}"
      header_done=1
    fi
    tail -n +2 "${per_case}" | sed "s|^|${case_label},${suite},|" >> "${combined}"
    cases=$((cases + 1))
  done < <(suite_cases "${suite}")
done

printf "finished_utc    %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${out_dir}/manifest.txt"

printf "\n== 汇总（%d 条用例）==\n" "${cases}"
cat "${combined}"
printf "\n输出目录: %s\n" "${out_dir}"
printf "  结果   %s\n" "${combined}"
printf "  环境   %s\n" "${out_dir}/manifest.txt"
