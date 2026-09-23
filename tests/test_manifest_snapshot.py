"""补丁包三个产物之间的一致性。

changed-files/ 是 apply 真正落盘的内容，manifest.json 是逐文件摘要（status / apply 靠它判断
分叉），聚合 .patch 只供审阅。三者不一致时，错误要等到别人 apply 时才暴露，所以在仓内就把
对齐关系钉住。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ninfer_ternary.manifest import (
    PatchManifest,
    changed_files_root,
    default_manifest_path,
    default_patch_path,
)
from ninfer_ternary.patchset import sha256_file

SNAPSHOT = changed_files_root()


@pytest.fixture(scope="module")
def manifest() -> PatchManifest:
    """载入仓内补丁清单。

    Returns:
        解析好的补丁清单。
    """
    return PatchManifest.load(default_manifest_path())


def test_manifest_digest_matches_snapshot(manifest: PatchManifest) -> None:
    """清单里的摘要必须等于快照中同名文件的摘要。"""
    mismatched = [
        entry.path for entry in manifest.files if sha256_file(SNAPSHOT / entry.path) != entry.sha256
    ]
    assert mismatched == []


def test_snapshot_holds_exactly_the_manifest_files(manifest: PatchManifest) -> None:
    """快照目录里既不该缺文件，也不该多出清单之外的文件。"""
    on_disk = {str(path.relative_to(SNAPSHOT)) for path in SNAPSHOT.rglob("*") if path.is_file()}
    assert on_disk == {entry.path for entry in manifest.files}


def test_status_matches_upstream_digest_presence(manifest: PatchManifest) -> None:
    """修改条目必须带上游摘要，新增条目必须不带。"""
    wrong = [
        entry.path
        for entry in manifest.files
        if (entry.sha256_upstream is None) != (entry.status == "added")
    ]
    assert wrong == []


def test_every_snapshot_path_is_covered_by_the_counts(manifest: PatchManifest) -> None:
    """清单自报的新增/修改计数必须与逐条状态一致。"""
    assert manifest.added_count == sum(1 for e in manifest.files if e.status == "added")
    assert manifest.modified_count == sum(1 for e in manifest.files if e.status == "modified")
    assert len(manifest.files) == manifest.added_count + manifest.modified_count


def test_aggregate_patch_labels_every_file(manifest: PatchManifest) -> None:
    """聚合 diff 必须给每个文件两侧都打上 a/ 与 b/ 标签，才能直接喂给 patch -p1。"""
    text = default_patch_path().read_text(encoding="utf-8")
    missing = [
        entry.path
        for entry in manifest.files
        if f"--- a/{entry.path}\n" not in text or f"+++ b/{entry.path}\n" not in text
    ]
    assert missing == []


def test_snapshot_is_not_empty() -> None:
    """快照目录必须存在且非空，否则上面的用例会以"空集合相等"的方式假通过。"""
    assert SNAPSHOT.is_dir()
    assert any(Path(SNAPSHOT).rglob("*"))
