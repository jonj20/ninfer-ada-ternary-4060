"""仓库卫生：缓存与字节码垃圾不得进入索引、历史或发布产物。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ninfer_ternary.manifest import repo_root

_GARBAGE_SEGMENTS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"})
_GARBAGE_SUFFIXES = (".pyc", ".pyo")
_REQUIRED_IGNORE_PATTERNS = ("__pycache__/", "*.py[oc]")


def _git(args: list[str], root: Path) -> list[str]:
    """执行 git 子命令并按行返回输出。

    Args:
        args: 传给 git 的参数，不含可执行文件名。
        root: 仓库根目录。

    Returns:
        去掉空行后的输出行。

    Raises:
        subprocess.CalledProcessError: git 以非零状态退出。
    """
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def _is_garbage(path: str) -> bool:
    """判断仓库内路径是否属于缓存或字节码垃圾。

    Args:
        path: 相对仓库根的路径。

    Returns:
        命中缓存目录名或字节码后缀时为 True。
    """
    if _GARBAGE_SEGMENTS & set(Path(path).parts):
        return True
    return path.endswith(_GARBAGE_SUFFIXES)


@pytest.fixture(name="git_root")
def fixture_git_root() -> Path:
    """提供可用的 git 仓库根目录，环境不满足时跳过用例。

    Returns:
        仓库根目录。
    """
    if shutil.which("git") is None:
        pytest.skip("环境缺少 git")
    root = repo_root()
    if not (root / ".git").exists():
        pytest.skip("当前不是 git 检出版本")
    return root


def test_gitignore_covers_cache_dirs(git_root: Path) -> None:
    """忽略规则必须覆盖缓存目录与字节码后缀。"""
    patterns = {
        line.strip()
        for line in (git_root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    missing = sorted(p for p in _REQUIRED_IGNORE_PATTERNS if p not in patterns)
    assert not missing, f"忽略规则缺少 {missing}"


def test_no_tracked_garbage(git_root: Path) -> None:
    """索引中不得存在缓存目录或字节码。"""
    offenders = sorted(p for p in _git(["ls-files"], git_root) if _is_garbage(p))
    assert not offenders, f"索引含垃圾路径 {offenders[:5]}"


def test_history_free_of_garbage(git_root: Path) -> None:
    """可达历史中不得存在缓存目录或字节码。

    只检查可达对象：不可达对象不会随克隆传播，检查它们会误报本地改写残留。
    """
    offenders = []
    for line in _git(["rev-list", "--objects", "--all"], git_root):
        fields = line.split(" ", 1)
        if len(fields) == 2 and _is_garbage(fields[1]):
            offenders.append(fields[1])
    assert not offenders, f"历史含垃圾路径 {offenders[:5]}"
