"""临时树的归属：全部落在项目专属根下，根目录可覆盖。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ninfer_ternary.engine import BuildRequest, scratch_parent
from ninfer_ternary.manifest import PatchManifest, default_manifest_path
from ninfer_ternary.scratch import TMP_ROOT_ENV, prune_temp_root, temp_root


@pytest.fixture(name="no_override")
def fixture_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉根目录覆盖变量，回到默认行为。"""
    monkeypatch.delenv(TMP_ROOT_ENV, raising=False)


def test_temp_root_defaults_under_system_tmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_override: None,
) -> None:
    """默认根目录是系统临时目录下的 ninfer-ternary，且会被创建。"""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    root = temp_root()
    assert root == tmp_path / "ninfer-ternary"
    assert root.is_dir()


def test_temp_root_honours_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NINFER_TERNARY_TMPDIR 直接决定根目录。"""
    override = tmp_path / "scratch"
    monkeypatch.setenv(TMP_ROOT_ENV, str(override))
    assert temp_root() == override
    assert override.is_dir()


def test_scratch_parent_defaults_to_temp_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_override: None,
) -> None:
    """请求未指定 scratch 时，临时树的父目录是项目临时根。"""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    request = BuildRequest.from_environment(PatchManifest.load(default_manifest_path()))
    assert request.scratch is None
    assert scratch_parent(request) == tmp_path / "ninfer-ternary"


def test_scratch_parent_keeps_explicit_dir(tmp_path: Path, no_override: None) -> None:
    """显式给出 scratch 时不覆盖，只补建目录。"""
    explicit = tmp_path / "somewhere"
    request = BuildRequest(
        repository="r",
        commit="c",
        scratch=explicit,
    )
    assert scratch_parent(request) == explicit
    assert explicit.is_dir()


def test_prune_temp_root_removes_empty_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_override: None,
) -> None:
    """默认根目录空了就删掉，不给系统临时目录留空壳。"""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    root = temp_root()
    assert root.is_dir()
    prune_temp_root()
    assert not root.exists()


def test_prune_temp_root_keeps_non_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_override: None,
) -> None:
    """根下还有临时树时不删，免得打断别的构建。"""
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    (temp_root() / "build-running").mkdir()
    prune_temp_root()
    assert (tmp_path / "ninfer-ternary" / "build-running").is_dir()


def test_prune_temp_root_keeps_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """显式覆盖的根目录归调用方管，不替人删。"""
    override = tmp_path / "scratch"
    monkeypatch.setenv(TMP_ROOT_ENV, str(override))
    temp_root()
    prune_temp_root()
    assert override.is_dir()
