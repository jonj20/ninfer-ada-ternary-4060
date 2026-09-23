#!/usr/bin/env bash
# 生成 text-only 三元 .ninfer（默认 PTQ1_0，裁 vision/mtp/dflash2）。
# Linux/4060 入口；内部转调 tools/pack_text.py。
#
# 用法:
#   tools/pack_text.sh --template /path/qwen3_8_27b-v2.ninfer \
#                      --gguf /path/Ternary-Bonsai-2-27B-PTQ1_0.gguf
#   tools/pack_text.sh --check-only --template ... --gguf ...
#   export NINFER_TERNARY_TEMPLATE=... NINFER_TERNARY_GGUF=...
#   tools/pack_text.sh
#
# NINFER_ROOT 可选：不设则用本仓 tools/artifact 快照。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

if [[ -z "${NINFER_ROOT:-}" && -d "$ROOT/tools/artifact" ]]; then
  export NINFER_ROOT="$ROOT"
fi

# 模板/GGUF：CLI 已带则不再要求环境变量
has_tpl=0
has_gguf=0
for a in "$@"; do
  [[ "$a" == "--template" || "$a" == --template=* ]] && has_tpl=1
  [[ "$a" == "--gguf" || "$a" == --gguf=* ]] && has_gguf=1
done
if [[ $has_tpl -eq 0 && -z "${NINFER_TERNARY_TEMPLATE:-}" ]]; then
  echo "缺少模板：--template <v2.ninfer> 或 NINFER_TERNARY_TEMPLATE" >&2
  exit 1
fi
if [[ $has_gguf -eq 0 && -z "${NINFER_TERNARY_GGUF:-}" ]]; then
  echo "缺少 GGUF：--gguf <PTQ1_0.gguf> 或 NINFER_TERNARY_GGUF" >&2
  exit 1
fi

exec "$PY" "$ROOT/tools/pack_text.py" "$@"
