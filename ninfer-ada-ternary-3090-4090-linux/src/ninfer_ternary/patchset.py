"""把补丁快照应用到一份 ninfer 检出，并报告每个文件的状态。

设计取舍：本仓发的是"整文件快照"而不是 diff，覆盖即可生效，因此不需要 patch 工具，
也不会因为上游空白差异而失败。代价是必须先确认目标树是纯净的 —— 这正是本模块用
上游摘要做前置校验的原因。
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .manifest import FileEntry, PatchManifest

_LOGGER = logging.getLogger(__name__)


class FileState(StrEnum):
    """单个文件相对补丁集合的状态。

    MISSING 只对 modified 条目有意义（上游应有的文件不见了）；added 条目缺失即为 PRISTINE。
    """

    PRISTINE = "pristine"
    PATCHED = "patched"
    DIVERGED = "diverged"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class FileResult:
    """单个文件的检查或应用结果。

    Attributes:
        entry: 清单中的文件条目。
        state: 该文件当前的状态。
    """

    entry: FileEntry
    state: FileState


@dataclass(frozen=True, slots=True)
class PatchReport:
    """整份补丁的检查或应用结果。

    Attributes:
        results: 逐文件结果。
        applied: 本次是否真正写入了文件。
    """

    results: tuple[FileResult, ...]
    applied: bool

    def count(self, state: FileState) -> int:
        """统计处于指定状态的文件数。

        Args:
            state: 目标状态。

        Returns:
            处于该状态的文件数量。
        """
        return sum(1 for item in self.results if item.state is state)

    @property
    def is_fully_applied(self) -> bool:
        """返回是否全部文件都已处于补丁后状态。"""
        return all(item.state is FileState.PATCHED for item in self.results)

    @property
    def is_pristine(self) -> bool:
        """返回是否全部文件都处于上游原状。"""
        return all(item.state is FileState.PRISTINE for item in self.results)


class PatchError(RuntimeError):
    """补丁无法安全应用。"""


def sha256_file(path: Path) -> str:
    """计算文件的 sha256 摘要。

    Args:
        path: 目标文件。

    Returns:
        十六进制摘要字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(repo: Path, entry: FileEntry) -> FileState:
    """判断目标树中某个文件的状态。

    Args:
        repo: ninfer 源码树根目录。
        entry: 清单条目。

    Returns:
        该文件的状态。
    """
    target = repo / entry.path
    if not target.is_file():
        # added 条目在上游本来就不存在，"文件缺失"对它是正常状态而非异常。
        return FileState.PRISTINE if entry.status == "added" else FileState.MISSING
    digest = sha256_file(target)
    if digest == entry.sha256:
        return FileState.PATCHED
    if entry.sha256_upstream is not None and digest == entry.sha256_upstream:
        return FileState.PRISTINE
    return FileState.DIVERGED


def inspect(repo: Path, manifest: PatchManifest) -> PatchReport:
    """只读检查目标树，不写入任何文件。

    Args:
        repo: ninfer 源码树根目录。
        manifest: 补丁清单。

    Returns:
        逐文件状态报告。

    Raises:
        PatchError: repo 不是目录。
    """
    if not repo.is_dir():
        raise PatchError(f"目标仓库目录不存在: {repo}")
    results = tuple(FileResult(entry, classify(repo, entry)) for entry in manifest.files)
    return PatchReport(results=results, applied=False)


def apply_patch_set(
    repo: Path,
    manifest: PatchManifest,
    snapshot_root: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> PatchReport:
    """把补丁快照覆盖到目标树。

    Args:
        repo: ninfer 源码树根目录。
        manifest: 补丁清单。
        snapshot_root: 补丁快照目录（patches/changed-files）。
        force: 为 True 时，即使某些文件已经分叉也继续覆盖。
        dry_run: 为 True 时只报告将要发生的动作，不写文件。

    Returns:
        应用结果报告。

    Raises:
        PatchError: 目标目录不存在、快照缺文件，或存在分叉且未指定 force。
    """
    if not repo.is_dir():
        raise PatchError(f"目标仓库目录不存在: {repo}")
    if not snapshot_root.is_dir():
        raise PatchError(f"补丁快照目录不存在: {snapshot_root}")

    before = inspect(repo, manifest)
    blocking = [item for item in before.results if item.state is FileState.DIVERGED]
    if blocking and not force:
        names = ", ".join(item.entry.path for item in blocking[:5])
        raise PatchError(
            f"{len(blocking)} 个文件与上游和本补丁都不一致（例如 {names}）；"
            "请先确认这些改动，或用 --force 覆盖。"
        )

    if dry_run:
        return PatchReport(results=before.results, applied=False)

    for item in before.results:
        source = snapshot_root / item.entry.path
        if not source.is_file():
            raise PatchError(f"补丁快照缺少文件: {source}")
        target = repo / item.entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        _LOGGER.info("覆盖: %s", item.entry.path)

    after = inspect(repo, manifest)
    if not after.is_fully_applied:
        broken = [item.entry.path for item in after.results if item.state is not FileState.PATCHED]
        raise PatchError(f"应用后仍有文件摘要不符: {', '.join(broken[:5])}")
    return PatchReport(results=after.results, applied=True)


def restore_upstream(repo: Path, manifest: PatchManifest, backup_root: Path) -> int:
    """把目标树中被本补丁修改过的文件回退为备份内容。

    Args:
        repo: ninfer 源码树根目录。
        manifest: 补丁清单。
        backup_root: 存放上游原文的目录，路径结构与 repo 一致。

    Returns:
        实际恢复的文件数。

    Raises:
        PatchError: 备份缺失。
    """
    restored = 0
    for entry in manifest.files:
        if entry.status != "modified":
            continue
        backup = backup_root / entry.path
        if not backup.is_file():
            raise PatchError(f"备份缺失: {backup}")
        shutil.copyfile(backup, repo / entry.path)
        restored += 1
        _LOGGER.info("回退: %s", entry.path)
    return restored
