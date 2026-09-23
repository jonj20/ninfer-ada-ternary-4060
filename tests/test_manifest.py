"""补丁清单的解析与校验。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ninfer_ternary.manifest import (
    ManifestError,
    PatchManifest,
    changed_files_root,
    default_manifest_path,
)


def test_builtin_manifest_matches_snapshot() -> None:
    """仓内清单必须与补丁快照逐文件一致。"""
    manifest = PatchManifest.load(default_manifest_path())
    root = changed_files_root()
    assert manifest.files, "清单不应为空"
    for entry in manifest.files:
        path = root / entry.path
        assert path.is_file(), f"快照缺少 {entry.path}"
    assert len(manifest.files) == len(manifest.by_path())


def test_manifest_rejects_missing_file(tmp_path: Path) -> None:
    """清单文件不存在时必须抛出 ManifestError。"""
    with pytest.raises(ManifestError):
        PatchManifest.load(tmp_path / "missing.json")


def test_manifest_rejects_bad_status(tmp_path: Path) -> None:
    """status 只允许 added / modified。"""
    payload = {
        "target": {"repository": "r", "commit": "c"},
        "port": {
            "source_repository": "s",
            "source_revision": "v",
            "source_baseline": "b",
        },
        "files": [
            {"path": "a.cpp", "sha256": "0" * 64, "status": "renamed", "sha256_upstream": None}
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError):
        PatchManifest.load(path)


def test_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    """重复路径必须被拒绝。"""
    entry = {"path": "a.cpp", "sha256": "0" * 64, "status": "added", "sha256_upstream": None}
    payload = {
        "target": {"repository": "r", "commit": "c"},
        "port": {"source_repository": "s", "source_revision": "v", "source_baseline": "b"},
        "files": [entry, entry],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError):
        PatchManifest.load(path)
