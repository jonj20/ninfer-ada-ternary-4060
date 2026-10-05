#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

for script in "$root"/*.sh; do
  bash -n "$script"
done

for windows_script in "$root"/*.bat "$root"/*.ps1; do
  # A launcher either mirrors a same-named .sh, or (the Windows bridges that hand over to WSL)
  # names the .sh it runs. Both are acceptable; a dangling reference is not.
  same_named="${windows_script%.*}.sh"
  if [[ -f "$same_named" ]]; then
    continue
  fi
  referenced=""
  while read -r candidate; do
    [[ -f "$root/$candidate" ]] && referenced="$candidate"
  done < <(grep -aoE '[A-Za-z0-9_.-]+\.sh' "$windows_script" | sort -u)
  if [[ -z "$referenced" ]]; then
    printf 'No Bash counterpart for %s (neither %s nor any .sh it names exists)\n' \
      "$(basename "$windows_script")" "$(basename "$same_named")" >&2
    exit 1
  fi
done

cat > "$tmp/ninfer-serve" <<'SERVER'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$NINFER_TEST_ARGS"
SERVER
chmod +x "$tmp/ninfer-serve"
touch "$tmp/qwen3_8_27b.ninfer" "$tmp/qwen3_6_35b_a3b.ninfer"

launchers=(
  'run-qwen38-c1.sh:qwen3_8_27b.ninfer:--max-context:65536'
  'run-qwen38-c8.sh:qwen3_8_27b.ninfer:--max-concurrency:8'
  'run-qwen38-vision.sh:qwen3_8_27b.ninfer:--vision:--spec'
  'run-qwen36-35b-vision.sh:qwen3_6_35b_a3b.ninfer:--vision:--no-thinking'
)
for entry in "${launchers[@]}"; do
  IFS=: read -r script model expected value <<< "$entry"
  args="$tmp/${script%.sh}.args"
  # Run through bash: these files are committed without the executable bit, so a direct
  # invocation would test the checkout's file modes instead of the script.
  NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$args" \
    bash "$root/$script" "$tmp/$model" >/dev/null
  grep -Fx -- "$expected" "$args" >/dev/null
  grep -Fx -- "$value" "$args" >/dev/null
done

mkdir -- "$tmp/bin" "$tmp/models"
cat > "$tmp/bin/curl" <<'CURL'
#!/usr/bin/env bash
while (( $# )); do
  if [[ "$1" == '--output' ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
: > "$output"
CURL
chmod +x "$tmp/bin/curl"
PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" "$root/download-qwen38.sh" >/dev/null
PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" "$root/download-qwen36-35b-vision.sh" >/dev/null
[[ -f "$tmp/models/qwen3_8_27b.ninfer" ]]
[[ -f "$tmp/models/qwen3_6_35b_a3b.ninfer" ]]

printf '%s\n' 'Linux script checks passed.'
