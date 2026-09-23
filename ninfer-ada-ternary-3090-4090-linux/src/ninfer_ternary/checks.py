"""不依赖 torch 的落地自检。

tools/artifact 的三元注册是打包器的硬前提，而它具有单一事实来源：引擎侧的
src/artifact/storage_layouts.cpp 里的 quant_geometry()。本模块只做文本级核对，
因此不需要安装 torch 就能在打补丁后立刻发现"注册漏了 / 几何写错了"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: 引擎侧每一组的平面字节数（基础平面, 高位平面），组宽固定 128。
EXPECTED_GEOMETRY = {
    "PTQ1_0_G128": (24, 2),
    "PQ2_0_G128": (32, 0),
}


@dataclass(frozen=True, slots=True)
class Finding:
    """一条自检结论。

    Attributes:
        path: 相关文件（相对 ninfer 根目录）。
        message: 人类可读的结论。
        ok: 是否通过。
    """

    path: str
    message: str
    ok: bool


def _read(path: Path) -> str | None:
    """读取文本文件。

    Args:
        path: 目标文件。

    Returns:
        文件内容；文件不存在时返回 None。
    """
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def check_engine_formats(ninfer_root: Path) -> list[Finding]:
    """核对引擎侧的 C++ 三元格式几何。

    Args:
        ninfer_root: ninfer 源码树根目录。

    Returns:
        逐项结论列表。
    """
    storage = ninfer_root / "src" / "artifact" / "storage_layouts.cpp"
    text = _read(storage)
    rel = "src/artifact/storage_layouts.cpp"
    if text is None:
        return [Finding(rel, "文件缺失", False)]
    findings: list[Finding] = []
    for name, (base, high) in EXPECTED_GEOMETRY.items():
        pattern = re.compile("NumericFormat::" + name + r":\s*\n\s*return \{(\d+), (\d+), (\d+)\};")
        match = pattern.search(text)
        if match is None:
            findings.append(Finding(rel, name + " 未注册几何", False))
            continue
        group, got_base, got_high = (int(match.group(i)) for i in (1, 2, 3))
        got = (group, got_base, got_high)
        want = (128, base, high)
        text_msg = name + " 几何 " + str(got)
        findings.append(
            Finding(rel, text_msg + ("" if got == want else "，期望 " + str(want)), got == want)
        )
    return findings


def check_artifact_registry(ninfer_root: Path) -> list[Finding]:
    """核对 Python 侧 tools/artifact 的三元注册。

    Args:
        ninfer_root: ninfer 源码树根目录。

    Returns:
        逐项结论列表。
    """
    numeric_rel = "tools/artifact/numeric.py"
    layouts_rel = "tools/artifact/layouts.py"
    numeric = _read(ninfer_root / numeric_rel)
    layouts = _read(ninfer_root / layouts_rel)
    if numeric is None:
        return [Finding(numeric_rel, "文件缺失", False)]
    if layouts is None:
        return [Finding(layouts_rel, "文件缺失", False)]

    findings: list[Finding] = []
    for name, (base, high) in EXPECTED_GEOMETRY.items():
        pattern = re.compile(
            "^" + name + r' = TernaryFormat\("' + name + r'", (\d+), (\d+), (\d+)\)',
            re.MULTILINE,
        )
        match = pattern.search(numeric)
        if match is None:
            findings.append(Finding(numeric_rel, name + " 未注册", False))
        else:
            group, got_base, got_high = (int(match.group(i)) for i in (1, 2, 3))
            got = (group, got_base, got_high)
            want = (128, base, high)
            findings.append(
                Finding(
                    numeric_rel,
                    name + " 平面字节 " + str(got) + ("" if got == want else "，期望 " + str(want)),
                    got == want,
                )
            )
        in_layout = '"' + name + '"' in layouts
        findings.append(
            Finding(
                layouts_rel,
                name
                + (" 已加入 row-split-k128-v1 格式集合" if in_layout else " 未加入布局格式集合"),
                in_layout,
            )
        )
    return findings


def check_ternary_sources(ninfer_root: Path) -> list[Finding]:
    """核对三元内核源文件是否齐全。

    Args:
        ninfer_root: ninfer 源码树根目录。

    Returns:
        逐项结论列表。
    """
    expected = (
        "src/ops/linear/ternary/ternary_dispatch.cpp",
        "src/ops/linear/ternary/ternary_dispatch.h",
        "src/ops/linear/ternary/ternary_launch.h",
        "src/ops/linear/ternary/ternary_rotation.cpp",
        "src/ops/linear/ternary/ternary_rotation.cu",
        "src/ops/linear/ternary/ternary_rotation.h",
        "src/ops/linear/ternary/ternary_rotation_kernels.cuh",
        "src/ops/linear/ternary/ternary_rowsplit_gemm.cu",
        "src/ops/linear/ternary/ternary_rowsplit_gemm.cuh",
        "src/ops/linear/ternary/ternary_rowsplit_gemv.cuh",
        "src/ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh",
        "src/ops/linear/ternary/ternary_rowsplit_storage.cuh",
        "src/ops/linear/ternary/ternary_row_view.h",
        "src/ops/kv_cache/hadamard_d256.cuh",
    )
    return [
        Finding(
            rel, "存在" if (ninfer_root / rel).is_file() else "缺失", (ninfer_root / rel).is_file()
        )
        for rel in expected
    ]


def check_all(ninfer_root: Path) -> list[Finding]:
    """执行全部自检。

    Args:
        ninfer_root: ninfer 源码树根目录。

    Returns:
        逐项结论列表。
    """
    return (
        check_ternary_sources(ninfer_root)
        + check_engine_formats(ninfer_root)
        + check_artifact_registry(ninfer_root)
    )
