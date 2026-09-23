#!/usr/bin/env python3
r"""生成 text-only 三元 `.ninfer` 制品（默认 PTQ1_0，裁 vision/mtp/dflash2）。

面向 RTX 4060 8G 文本路径：先 `pack.py check`，再 `pack.py build`，并校验产物
magic / identity / 对象前缀。路径可用环境变量或命令行覆盖；本机默认路径已写在
脚本常量里（TEMPLATE/GGUF/OUT_DIR），无参即可跑。

默认裁掉 vision/*、mtp/*、dflash2/*（显存紧、本期不做 DFlash 投机）；
需要保留某段时用 --keep-vision / --keep-mtp / --keep-dflash2。

用法::

    # 本机默认路径，一行
    python tools/pack_text.py

    # 只自检
    python tools/pack_text.py --check-only

    # 覆盖路径（Linux/4060）
    python tools/pack_text.py \
      --template /path/qwen3_8_27b-v2.ninfer \
      --gguf /path/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
      --out /path/out.ninfer

环境变量（命令行优先）：
  PYTHON                   打包用解释器（默认优先 D:\David\python\python.exe，需 ≥3.10）
  NINFER_ROOT              可选：完整 ninfer 检出；不设则用本仓 tools/artifact 快照
  NINFER_TERNARY_TEMPLATE  覆盖默认模板路径
  NINFER_TERNARY_GGUF      覆盖默认 GGUF 路径
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

# 打包用根：默认本仓（tools/artifact 快照已在仓内）；可用 NINFER_ROOT 指完整检出。
_DEFAULT_NINFER_ROOT = ""  # 空 = 用仓根
_DEFAULT_TEMPLATE = r"D:\LLM\llama\qwen3_8_27b-v2.ninfer"
_DEFAULT_GGUF = r"D:\LLM\llama\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
_DEFAULT_OUT_DIR = r"D:\LLM\ninfer-out"

_KIND_FILES = {
    "PTQ1_0": "Ternary-Bonsai-2-27B-PTQ1_0",
    "PQ2_0": "Ternary-Bonsai-2-27B-PQ2_0",
}


def _repo_tools() -> Path:
    return Path(__file__).resolve().parent


def _pack_py() -> Path:
    return _repo_tools() / "pack.py"


def _python_version(executable: str) -> tuple[int, int] | None:
    """读解释器主次版本；失败返回 None。"""
    try:
        out = subprocess.check_output(
            [executable, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
        major, minor = out.split()[:2]
        return int(major), int(minor)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _pick_python(preferred: str | None) -> str:
    """选一个能跑 pack 的解释器（需 ≥3.10，因 typing.TypeAlias 等）。

    Args:
        preferred: 用户指定的解释器；可用则直接返回。

    Returns:
        可执行的 Python 路径。

    Raises:
        SystemExit: 找不到满足版本要求的解释器。
    """
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    env_py = os.environ.get("PYTHON")
    if env_py:
        candidates.append(env_py)
    # 本机优先：D:\David\python 为 3.12（pack 需 TypeAlias）
    candidates.append(r"D:\David\python\python.exe")
    candidates.append(sys.executable)
    candidates.extend([
        r"C:\Users\j\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe",
        "python3",
        "python",
    ])

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        ver = _python_version(cand)
        if ver is not None and ver >= (3, 10):
            return cand

    raise SystemExit(
        "找不到 Python >= 3.10（pack 依赖 typing.TypeAlias 等）。\n"
        "  请设置 PYTHON=<解释器路径>，或安装 3.12 后重试。\n"
        f"  当前 sys.executable={sys.executable} version={sys.version_info[:2]}"
    )


def _resolve(cli: str | None, env: str, default: str) -> str:
    raw = cli or os.environ.get(env) or default
    return str(Path(raw).expanduser())


def _preflight(path: str, label: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"{label}不存在: {p}\n"
            f"  请用命令行参数或对应环境变量指定正确路径。"
        )


def _validate_artifact(path: Path, expect_no: tuple[str, ...]) -> dict:
    """读容器目录，校验 magic/identity/对象前缀；返回摘要 dict。"""
    with path.open("rb") as f:
        prefix = f.read(16)
        if len(prefix) != 16:
            raise SystemExit(f"制品过短，读不到前缀: {path}")
        magic, dlen = struct.unpack("<8sQ", prefix)
        if magic != b"NINFER\x00\x02":
            raise SystemExit(f"制品 magic 不是 NINFER v2: {magic!r}")
        blob = f.read(dlen)
        if len(blob) != dlen:
            raise SystemExit(f"制品目录 JSON 被截断: {path}")

    obj = json.loads(blob.decode("utf-8"))
    identity = obj.get("identity") or {}
    objects = obj.get("objects") or []
    if identity.get("weights_id") != "folded-ternary":
        raise SystemExit(
            f"identity.weights_id 应为 folded-ternary，实际 {identity.get('weights_id')!r}"
        )

    prefixes = Counter((o.get("name") or "?").split("/")[0] for o in objects)
    for bad in expect_no:
        if prefixes.get(bad):
            raise SystemExit(f"产物仍含 {bad}/* x{prefixes[bad]}，与 skip 开关不符")

    end = max((o["offset"] + o["bytes"] for o in objects), default=0)
    payload_off = (16 + dlen + 4095) // 4096 * 4096
    size = path.stat().st_size
    need = payload_off + end
    if size < need:
        raise SystemExit(f"产物不完整: size={size} < need={need}")

    return {
        "path": str(path),
        "bytes": size,
        "identity": identity,
        "n_objects": len(objects),
        "prefixes": dict(prefixes),
        "complete": size == need,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="check + build text-only 三元 .ninfer"
        "（默认 PTQ1_0，裁 vision/mtp/dflash2）",
    )
    ap.add_argument("--kind", choices=sorted(_KIND_FILES), default="PTQ1_0")
    ap.add_argument("--ninfer-root",
                    help="含 tools/artifact 的 ninfer 树；默认本仓根（内置快照）")
    ap.add_argument("--template", help="groupwise-int v2 模板 .ninfer")
    ap.add_argument("--gguf", help="三元 GGUF")
    ap.add_argument("--out", help="输出 .ninfer 路径（默认 out_dir/<kind>-text.ninfer）")
    ap.add_argument("--out-dir", help=f"输出目录（默认 {_DEFAULT_OUT_DIR}）")
    ap.add_argument("--check-only", action="store_true", help="只跑 pack check")
    ap.add_argument("--skip-check", action="store_true", help="跳过 check 直接 build")
    ap.add_argument("--skip-vision", action="store_true", default=True,
                    help="裁 vision/*（默认开）")
    ap.add_argument("--keep-vision", action="store_true", help="保留 vision/*")
    ap.add_argument("--skip-mtp", action="store_true", default=True,
                    help="裁 mtp/*（默认开）")
    ap.add_argument("--keep-mtp", action="store_true", help="保留 mtp/*")
    ap.add_argument("--skip-dflash2", action="store_true", default=True,
                    help="裁 dflash2/*（默认开：8G 显存紧；不用 DFlash 投机）")
    ap.add_argument("--keep-dflash2", action="store_true", help="保留 dflash2/*")
    ap.add_argument("--python",
                    help="打包用解释器；默认自动选 >=3.10（PYTHON 环境变量亦可）")
    args = ap.parse_args()

    pack_python = _pick_python(args.python)

    skip_vision = args.skip_vision and not args.keep_vision
    skip_mtp = args.skip_mtp and not args.keep_mtp
    skip_dflash2 = args.skip_dflash2 and not args.keep_dflash2

    # 空默认 → 仓根；bootstrap 在无 NINFER_ROOT 时也会回退到仓内 tools/artifact
    ninfer_root = _resolve(
        args.ninfer_root,
        "NINFER_ROOT",
        _DEFAULT_NINFER_ROOT or str(_repo_tools().parent),
    )
    template = _resolve(args.template, "NINFER_TERNARY_TEMPLATE", _DEFAULT_TEMPLATE)
    gguf = _resolve(
        args.gguf,
        "NINFER_TERNARY_GGUF",
        str(Path(_DEFAULT_GGUF).parent / f"{_KIND_FILES[args.kind]}.gguf"),
    )
    out_dir = Path(args.out_dir or _DEFAULT_OUT_DIR).expanduser()
    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        suffix = "-text" if (skip_vision or skip_mtp or skip_dflash2) else ""
        out_path = out_dir / f"{_KIND_FILES[args.kind]}{suffix}.ninfer"

    _preflight(ninfer_root, "NINFER_ROOT")
    _preflight(str(Path(ninfer_root) / "tools" / "artifact"), "tools/artifact")
    _preflight(template, "模板")
    _preflight(gguf, "GGUF")
    _preflight(str(_pack_py()), "pack.py")

    env = os.environ.copy()
    env["NINFER_ROOT"] = ninfer_root
    env["NINFER_TERNARY_TEMPLATE"] = template
    env["NINFER_TERNARY_GGUF"] = gguf
    env["NINFER_TERNARY_SKIP_VISION"] = "1" if skip_vision else "0"
    env["NINFER_TERNARY_SKIP_MTP"] = "1" if skip_mtp else "0"
    env["NINFER_TERNARY_SKIP_DFLASH2"] = "1" if skip_dflash2 else "0"

    filters = []
    if skip_vision:
        filters.append("vision")
    if skip_mtp:
        filters.append("mtp")
    if skip_dflash2:
        filters.append("dflash2")
    print(f"kind       : {args.kind}")
    print(f"python     : {pack_python}")
    print(f"NINFER_ROOT: {ninfer_root}")
    print(f"template   : {template}")
    print(f"gguf       : {gguf}")
    print(f"out        : {out_path}")
    print(f"filters    : omit {', '.join(filters) if filters else '(none)'}")

    base_cmd = [pack_python, str(_pack_py()), "--template", template, "--gguf", gguf]
    if skip_vision:
        base_cmd.append("--skip-vision")
    if skip_mtp:
        base_cmd.append("--skip-mtp")
    if skip_dflash2:
        base_cmd.append("--skip-dflash2")

    if not args.skip_check:
        print("\n=== pack check ===")
        rc = subprocess.call(base_cmd + ["check"], env=env)
        if rc != 0:
            raise SystemExit(f"pack check 失败，exit={rc}")
        if args.check_only:
            print("\nRESULT: check-only OK")
            return 0
    elif args.check_only:
        raise SystemExit("--check-only 与 --skip-check 互斥")

    if out_path.exists():
        raise SystemExit(
            f"拒绝覆盖已存在制品: {out_path}\n"
            f"  请删除旧文件或改 --out。"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== pack build ===")
    rc = subprocess.call(base_cmd + ["build", str(out_path)], env=env)
    if rc != 0:
        raise SystemExit(f"pack build 失败，exit={rc}")

    expect_no = tuple(filters)
    summary = _validate_artifact(out_path, expect_no)
    print("\n=== artifact ===")
    print(f"  path      : {summary['path']}")
    print(f"  bytes     : {summary['bytes']:,} = {summary['bytes'] / 2**30:.3f} GiB")
    print(f"  identity  : {summary['identity']}")
    print(f"  objects   : {summary['n_objects']}")
    print(f"  prefixes  : {summary['prefixes']}")
    print(f"  complete  : {summary['complete']}")
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
