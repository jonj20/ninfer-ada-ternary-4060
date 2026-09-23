"""补丁应用与状态判定。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ninfer_ternary.manifest import FileEntry, PatchManifest
from ninfer_ternary.patchset import (
    FileState,
    PatchError,
    apply_patch_set,
    classify,
    inspect,
    sha256_file,
)


def _digest(text: str) -> str:
    """返回字符串的 sha256，用于构造测试条目。"""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest(entries: list[FileEntry]) -> PatchManifest:
    """构造只带指定文件的清单。"""
    return PatchManifest(
        target_repository="repo",
        target_commit="commit",
        source_repository="src",
        source_revision="rev",
        source_baseline="base",
        files=tuple(entries),
    )


@pytest.fixture()
def layout(tmp_path: Path) -> tuple[Path, Path, PatchManifest]:
    """构造最小的"上游 + 快照"目录对。"""
    repo = tmp_path / "repo"
    snapshot = tmp_path / "snapshot"
    repo.mkdir()
    snapshot.mkdir()
    (repo / "a.txt").write_text("old-a\n", encoding="utf-8")
    (snapshot / "a.txt").write_text("new-a\n", encoding="utf-8")
    (snapshot / "b.txt").write_text("new-b\n", encoding="utf-8")
    manifest = _manifest(
        [
            FileEntry("a.txt", _digest("new-a\n"), "modified", _digest("old-a\n")),
            FileEntry("b.txt", _digest("new-b\n"), "added", None),
        ]
    )
    return repo, snapshot, manifest


def test_classify_states(layout: tuple[Path, Path, PatchManifest]) -> None:
    """三种状态判定必须准确。"""
    repo, _snapshot, manifest = layout
    assert classify(repo, manifest.files[0]) is FileState.PRISTINE
    # added 条目在上游不存在，缺失即为"未打补丁"的原状
    assert classify(repo, manifest.files[1]) is FileState.PRISTINE
    (repo / "a.txt").write_text("changed\n", encoding="utf-8")
    assert classify(repo, manifest.files[0]) is FileState.DIVERGED


def test_apply_then_status(layout: tuple[Path, Path, PatchManifest]) -> None:
    """应用后必须全部为 patched，且摘要与清单一致。"""
    repo, snapshot, manifest = layout
    before = inspect(repo, manifest)
    assert before.is_pristine is True
    assert before.count(FileState.PRISTINE) == 2

    report = apply_patch_set(repo, manifest, snapshot)
    assert report.applied is True
    assert report.is_fully_applied is True
    assert sha256_file(repo / "a.txt") == manifest.files[0].sha256


def test_dry_run_writes_nothing(layout: tuple[Path, Path, PatchManifest]) -> None:
    """dry-run 不得改动文件。"""
    repo, snapshot, manifest = layout
    apply_patch_set(repo, manifest, snapshot, dry_run=True)
    assert (repo / "a.txt").read_text(encoding="utf-8") == "old-a\n"
    assert not (repo / "b.txt").exists()


def test_diverged_blocks_without_force(layout: tuple[Path, Path, PatchManifest]) -> None:
    """分叉文件在未指定 force 时必须阻断。"""
    repo, snapshot, manifest = layout
    (repo / "a.txt").write_text("local\n", encoding="utf-8")
    with pytest.raises(PatchError):
        apply_patch_set(repo, manifest, snapshot)
    apply_patch_set(repo, manifest, snapshot, force=True)
    assert (repo / "a.txt").read_text(encoding="utf-8") == "new-a\n"


def test_missing_snapshot_file_raises(layout: tuple[Path, Path, PatchManifest]) -> None:
    """快照缺文件时必须报错，而不是静默跳过。"""
    repo, snapshot, manifest = layout
    (snapshot / "b.txt").unlink()
    with pytest.raises(PatchError):
        apply_patch_set(repo, manifest, snapshot)
