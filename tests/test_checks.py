"""落地自检与 tools/artifact 三元注册。

本模块的用例全部只用 pytest 的临时目录与仓内快照，不依赖机器上恰好存在某个
ninfer 检出、也不依赖该检出恰好处于某个状态：快照 `patches/changed-files/`
本身就是"已打补丁"的权威内容，把它铺开就是正控树，不放三元件就是负控树。
需要对着真实检出核对的用例，统一由 `NIFER_TERNARY_REPO` 显式开启，未设则跳过。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

from ninfer_ternary.checks import check_all
from ninfer_ternary.manifest import changed_files_root

#: 仓内"已打补丁"快照，同时充当正控树的素材来源。
SNAPSHOT = changed_files_root()

#: 指向一个真实的、已打补丁的 ninfer 检出；未设置时相关用例跳过。
REPO_ENV = "NIFER_TERNARY_REPO"


def _repo_from_env() -> Path | None:
    """读取真实检出路径。

    Returns:
        `NIFER_TERNARY_REPO` 指向的目录；未设置或不是目录时返回 None。
    """
    value = os.environ.get(REPO_ENV)
    if not value:
        return None
    path = Path(value)
    return path if path.is_dir() else None


def _require_repo() -> Path:
    """返回真实检出根目录；未设置时跳过当前用例。

    Returns:
        已打补丁的 ninfer 检出根目录。
    """
    repo = _repo_from_env()
    if repo is None:
        pytest.skip(f"未设置 {REPO_ENV}")
    assert repo is not None
    return repo


def _patched_tree(tmp_path: Path) -> Path:
    """把仓内快照铺成一棵最小的 ninfer 树。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        铺好三元件与格式注册的树根。
    """
    root = tmp_path / "patched"
    shutil.copytree(SNAPSHOT, root)
    return root


def _upstream_tree(tmp_path: Path) -> Path:
    """构造一棵没有三元注册、没有三元内核的未打补丁树。

    只保留自检真正会读的文件，形状对齐上游：格式枚举到 Q4 为止，也没有
    `ops/linear/ternary/` 与 `ops/kv_cache/`。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        未打补丁的树根。
    """
    root = tmp_path / "upstream"
    (root / "src" / "artifact").mkdir(parents=True)
    (root / "tools" / "artifact").mkdir(parents=True)
    (root / "src" / "artifact" / "storage_layouts.cpp").write_text(
        "Geometry quant_geometry(NumericFormat fmt) {\n"
        "    switch (fmt) {\n"
        "    case NumericFormat::Q4G64_F16S:\n"
        "        return {64, 36, 0};\n"
        "    default:\n"
        '        throw std::invalid_argument("unsupported");\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "tools" / "artifact" / "numeric.py").write_text(
        'Q4G64_F16S = QuantFormat("Q4G64_F16S", 64, 4)\n',
        encoding="utf-8",
    )
    (root / "tools" / "artifact" / "layouts.py").write_text(
        'Q4G64_F16S = LayoutSpec("row-split-k128-v1", ("Q4G64_F16S",))\n',
        encoding="utf-8",
    )
    return root


def test_checks_fail_on_tree_without_ternary(tmp_path: Path) -> None:
    """负控：缺三元注册与内核的树必须被自检判为失败。"""
    findings = check_all(_upstream_tree(tmp_path))
    assert findings, "自检不应为空"
    assert any(not finding.ok for finding in findings)


def test_checks_pass_on_snapshot_tree(tmp_path: Path) -> None:
    """正控：快照铺开的树必须全绿。"""
    findings = check_all(_patched_tree(tmp_path))
    failed = [finding for finding in findings if not finding.ok]
    assert not failed, failed


def test_checks_report_missing_files(tmp_path: Path) -> None:
    """文件缺失必须逐条报出，而不是静默通过。"""
    root = tmp_path / "empty"
    root.mkdir()
    findings = check_all(root)
    assert findings
    assert all(not finding.ok for finding in findings)


def _torch_available() -> bool:
    """返回当前解释器是否可导入 torch。

    Returns:
        可导入为 True。
    """
    return importlib.util.find_spec("torch") is not None


def test_checks_on_real_checkout() -> None:
    """可选：对真实检出跑一遍自检，需 NIFER_TERNARY_REPO 指向已打补丁的树。"""
    failed = [finding for finding in check_all(_require_repo()) if not finding.ok]
    assert not failed, failed


@pytest.mark.skipif(not _torch_available(), reason="需要 torch 才能导入 tools/artifact")
def test_artifact_geometry_matches_engine() -> None:
    """tools/artifact 的三元几何必须与引擎侧数值一致，需 NIFER_TERNARY_REPO。"""
    repo = _require_repo()

    sys.path.insert(0, str(repo))
    try:
        from tools.artifact import (
            ROW_SPLIT_K128_V1,
            encode_row_split,
            encoded_size,
            row_split_geometry,
        )
    finally:
        sys.path.pop(0)

    expected = {"PTQ1_0_G128": 278_118_400, "PQ2_0_G128": 337_715_200}
    for name, want in expected.items():
        assert name in ROW_SPLIT_K128_V1.formats
        geometry = row_split_geometry(name, (248320, 5120))
        assert geometry.payload_bytes == want
        assert encoded_size("row-split-k128-v1", name, (248320, 5120)) == want
    with pytest.raises(ValueError):
        encode_row_split(None, None, "PQ2_0_G128", (248320, 5120))
