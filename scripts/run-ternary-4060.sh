#!/usr/bin/env bash
# 启动 Ternary Bonsai 2 27B（PTQ1_0）的 ninfer-serve，自带 llama.cpp 那套 WebUI。
#
# 用法：
#   scripts/run-ternary-4060.sh [模型路径] [额外传给 ninfer-serve 的参数...]
#
# 环境变量（都有默认值，一般不用改）：
#   NINFER_MODE     think | no-think，默认 no-think。会套用模型卡推荐的那套采样参数（见下）
#   NINFER_SERVER   ninfer-serve 可执行文件；不给就按 仓根/build_4060 → /home/jon/build_4060b → PATH 探测
#   NINFER_MODEL    模型制品；不给就用 $1，再不给用下面那个 4060 专用默认路径
#   NINFER_HOST     监听地址，默认 0.0.0.0（Windows 侧浏览器直接访问 localhost 即可）
#   NINFER_PORT     端口，默认 8080
#   NINFER_CTX      上下文上限，默认 48000。8G 卡上 4bit KV 系列实测上限是 49152（再大就装不下），
#                   48000 已经很靠边：启动后只剩约 100 MiB。要宽松些用 32768（剩 400 MiB）。
#   NINFER_KV       KV 量化布局，默认 rk4v4-e8
#   NINFER_CONC     并发解码槽，默认 1
#   NINFER_MAXTOK   客户端没给 max_tokens 时的默认值，默认 48000——也就是"不设限"，让模型自己写完。
#                   实际能生成多少由上下文剩余量决定（总预算 = prompt + 输出）。WebUI 不发
#                   max_tokens，全靠这个兜底：曾经默认 512，列表类回答被 finish=output_limit 砍断。
#   NINFER_LOG      设了就把服务输出写到这个文件而不是打在终端
#
# 采样参数取自模型卡的推荐值，客户端没显式指定时由服务端兜底：
#   思考模式     temperature 1.0  top_p 0.95  top_k 20  min_p 0.05  presence_penalty 0.0
#   指令/非思考  temperature 0.7  top_p 0.80  top_k 20  min_p 0.0   presence_penalty 1.5
# 两套的 repetition_penalty 都是 1.0，也就是不做重复惩罚——引擎没有独立的 repetition-penalty
# 开关（只有语义不同的 --frequency-penalty），所以这一项不需要传参，保持默认即可。
#
# Ctrl+C 停服务。
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# 可执行文件探测：显式指定 → 仓内构建 / 4060 专用构建目录 / PATH 里所有可用的，取最新的那个。
# 这台机器上两个构建目录并存（仓内 build_4060 是旧的），选新的才不会给出偏慢的性能数字。
server="${NINFER_SERVER:-}"
if [[ -z "$server" ]]; then
  candidates=(
    "$root/build_4060/apps/ninfer-serve"
    "/home/jon/build_4060b/apps/ninfer-serve"
  )
  path_hit="$(command -v ninfer-serve 2>/dev/null || true)"
  [[ -n "$path_hit" ]] && candidates+=("$path_hit")
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      if [[ -z "$server" || "$candidate" -nt "$server" ]]; then
        server="$candidate"
      fi
    fi
  done
fi

model="${1:-${NINFER_MODEL:-/mnt/e/gguf/qwen3.6-35b-a3b/Ternary-Bonsai-2-27B-PTQ1_0-text.ninfer}}"
[[ $# -gt 0 ]] && shift

if [[ -z "$server" || ! -x "$server" ]]; then
  printf '找不到 ninfer-serve（试过仓内 build_4060 与 /home/jon/build_4060b）\n' >&2
  printf '先构建：ninja -C /home/jon/build_4060b apps/ninfer-serve，或用 NINFER_SERVER 指定\n' >&2
  exit 2
fi
if [[ ! -f "$model" ]]; then
  printf '找不到模型制品：%s\n' "$model" >&2
  printf '用第一个位置参数或 NINFER_MODEL 指定 .ninfer 文件\n' >&2
  exit 2
fi

# 二进制早于最新源码说明是没重新构建，跑出来的性能数字会让人误解
newest_src="$(find "$root/src" -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' 2>/dev/null | sort | tail -1)"
if [[ -n "$newest_src" && "$server" -ot "$newest_src" ]]; then
  printf '提醒：%s 比最新的源码还旧，结果会偏慢\n' "$server"
  printf '重新构建：ninja -C %s apps/ninfer-serve\n' "$(dirname "$(dirname "$server")")"
fi

host="${NINFER_HOST:-0.0.0.0}"
port="${NINFER_PORT:-8080}"
ctx="${NINFER_CTX:-48000}"
kv="${NINFER_KV:-rk4v4-e8}"
conc="${NINFER_CONC:-1}"
mode="${NINFER_MODE:-think}"
# 输出上限默认不设限（48000），实际由上下文剩余量决定；客户端显式传 max_tokens 时以客户端为准。
maxtok="${NINFER_MAXTOK:-48000}"

case "$mode" in
  think)
    sampler=(--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 --presence-penalty 0.0)
    ;;
  no-think)
    sampler=(--temperature 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --presence-penalty 1.5)
    ;;
  *)
    printf 'NINFER_MODE 只能是 think 或 no-think，收到：%s\n' "$mode" >&2
    exit 2
    ;;
esac

printf '可执行文件 : %s\n' "$server"
printf '模型       : %s\n' "$model"
printf '模式       : %s（%s）\n' "$mode" "$([[ $mode == think ]] && echo '会输出思考内容' || echo '不输出思考内容')"
printf '采样参数   : temperature %s  top_p %s  top_k %s  min_p %s  presence_penalty %s\n' \
  "${sampler[1]}" "${sampler[3]}" "${sampler[5]}" "${sampler[7]}" "${sampler[9]}"
printf '上下文     : %s（KV %s，并发 %s）\n' "$ctx" "$kv" "$conc"
printf '输出上限   : %s token（客户端没给 max_tokens 时用这个；实际能生成多少由上下文剩余量决定）\n' "$maxtok"
printf 'WebUI      : http://localhost:%s/\n' "$port"
printf '接口       : http://localhost:%s/v1（chat/completions、responses、messages）\n' "$port"
printf '装载权重约 60-70 秒，之后 prefill 约 125 tok/s、decode 约 22.6 tok/s\n'
if [[ "$ctx" -ge 45000 ]]; then
  printf '注意       : %s 上下文已经很靠边（4bit KV 实测上限 49152），启动后只剩约 100 MiB，\n' "$ctx"
  printf '             多轮长对话有 OOM 风险；想要宽松些就 NINFER_CTX=32768（剩 400 MiB）\n'
fi
printf '停止       : Ctrl+C\n'
printf '\n冒烟请求：\n'
printf '  curl http://localhost:%s/v1/chat/completions -H "Content-Type: application/json" \\\n' "$port"
printf '    -d '"'"'{"model":"local","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'"'"'\n\n'

args=(
  "$model"
  --host "$host"
  --port "$port"
  --max-context "$ctx"
  --kv-dtype "$kv"
  --max-concurrency "$conc"
  --default-max-tokens "$maxtok"
)

# 思考模式不关思考；非思考模式按模型卡的指令模式关掉，并可用 NINFER_REASONING_EFFORT 选深度
if [[ "$mode" == "think" ]]; then
  args+=(--reasoning-effort "${NINFER_REASONING_EFFORT:-medium}")
else
  args+=(--no-thinking)
fi
args+=("${sampler[@]}")

if [[ -n "${NINFER_LOG:-}" ]]; then
  exec "$server" "${args[@]}" "$@" >"$NINFER_LOG" 2>&1
else
  exec "$server" "${args[@]}" "$@"
fi
