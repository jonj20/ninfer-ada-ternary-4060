"""本仓补丁清单的模型与读取。

manifest.json 记录"本仓的改动是叠在哪一棵 ninfer 树的哪个提交上"，以及每个文件的
上游摘要与改动后摘要。有了这两组摘要，就能把任意一份检出分类成
"未改动 / 已打本补丁 / 已经分叉"三种状态，而不必重新 diff。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from . import assets

MANIFEST_NAME = "manifest.json"
CHANGED_FILES_DIR = "changed-files"
#: 聚合 diff 的文件名。它只供审阅，但审阅者会按名字引用它，所以名字固定在这里。
AGGREGATE_PATCH_NAME = "0001-ternary-port-on-ninfer-4090.patch"


class ManifestError(ValueError):
    """manifest.json 不满足本仓约定的结构。"""


@dataclass(frozen=True, slots=True)
class FileEntry:
    """补丁集合中的一个文件。

    Attributes:
        path: 相对 ninfer 源码树根目录的路径（POSIX 分隔符）。
        sha256: 打完补丁后的文件摘要。
        status: added（上游没有该文件）或 modified。
        sha256_upstream: 上游原文摘要；added 时为 None。
    """

    path: str
    sha256: str
    status: str
    sha256_upstream: str | None


@dataclass(frozen=True, slots=True)
class PatchManifest:
    """整份补丁清单。

    Attributes:
        target_repository: 目标 ninfer 仓库地址。
        target_commit: 目标提交（补丁所基于的版本）。
        source_repository: 改动来源仓库。
        source_revision: 改动来源版本说明。
        source_baseline: 原始补丁所基于的基座树。
        files: 全部文件条目。
    """

    target_repository: str
    target_commit: str
    source_repository: str
    source_revision: str
    source_baseline: str
    files: tuple[FileEntry, ...]

    @property
    def added_count(self) -> int:
        """返回新增文件数。"""
        return sum(1 for entry in self.files if entry.status == "added")

    @property
    def modified_count(self) -> int:
        """返回修改文件数。"""
        return sum(1 for entry in self.files if entry.status == "modified")

    def by_path(self) -> Mapping[str, FileEntry]:
        """返回按路径索引的只读映射。

        Returns:
            path 到 FileEntry 的只读映射。
        """
        return MappingProxyType({entry.path: entry for entry in self.files})

    @classmethod
    def load(cls, path: Path) -> "PatchManifest":
        """从 JSON 文件读取清单。

        Args:
            path: manifest.json 的路径。

        Returns:
            解析后的清单。

        Raises:
            ManifestError: 文件缺失或字段不符合约定。
        """
        if not path.is_file():
            raise ManifestError(f"找不到补丁清单: {path}")
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ManifestError("manifest.json 顶层必须是对象")
        target = raw.get("target")
        port = raw.get("port")
        entries = raw.get("files")
        if (
            not isinstance(target, dict)
            or not isinstance(port, dict)
            or not isinstance(entries, list)
        ):
            raise ManifestError("manifest.json 缺少 target / port / files 段")
        files = tuple(_parse_entry(item) for item in entries)
        if len({entry.path for entry in files}) != len(files):
            raise ManifestError("manifest.json 存在重复路径")
        return cls(
            target_repository=_require_str(target, "repository"),
            target_commit=_require_str(target, "commit"),
            source_repository=_require_str(port, "source_repository"),
            source_revision=_require_str(port, "source_revision"),
            source_baseline=_require_str(port, "source_baseline"),
            files=files,
        )


def _require_str(block: Mapping[str, Any], key: str) -> str:
    """取出对象中的非空字符串字段。

    Args:
        block: 待检查的对象。
        key: 字段名。

    Returns:
        字段值。

    Raises:
        ManifestError: 字段缺失或不是非空字符串。
    """
    value = block.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"字段 {key} 必须是非空字符串")
    return value


def _parse_entry(item: Any) -> FileEntry:
    """把一条 JSON 记录解析为 FileEntry。

    Args:
        item: files 数组中的一项。

    Returns:
        解析后的条目。

    Raises:
        ManifestError: 结构不符合约定。
    """
    if not isinstance(item, dict):
        raise ManifestError("files 的每一项必须是对象")
    path = item.get("path")
    digest = item.get("sha256")
    status = item.get("status")
    upstream = item.get("sha256_upstream")
    if not isinstance(path, str) or not path:
        raise ManifestError("文件条目的 path 必须是非空字符串")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ManifestError(f"{path}: sha256 必须是 64 位十六进制")
    if status not in ("added", "modified"):
        raise ManifestError(f"{path}: status 只能是 added 或 modified")
    if status == "modified" and (not isinstance(upstream, str) or len(upstream) != 64):
        raise ManifestError(f"{path}: modified 条目必须带 sha256_upstream")
    if status == "added" and upstream is not None:
        raise ManifestError(f"{path}: added 条目不应带 sha256_upstream")
    return FileEntry(path=path, sha256=digest, status=status, sha256_upstream=upstream)


def repo_root() -> Path:
    """返回本仓根目录。

    Returns:
        本仓根目录的绝对路径；安装形态下该路径无意义，补丁数据请走下面的助手。
    """
    return assets.repo_root()


def patches_root() -> Path:
    """返回补丁数据目录。

    Returns:
        补丁目录路径：安装形态取随包副本，仓库形态取仓库里的 patches/。
    """
    return assets.patches_root()


def default_manifest_path() -> Path:
    """返回内置补丁清单的路径。

    Returns:
        manifest.json 的绝对路径。
    """
    return patches_root() / MANIFEST_NAME


def changed_files_root() -> Path:
    """返回补丁文件快照目录。

    Returns:
        changed-files 的绝对路径。
    """
    return patches_root() / CHANGED_FILES_DIR


def default_patch_path() -> Path:
    """返回聚合 diff 的路径。

    Returns:
        patches/ 下聚合补丁的绝对路径。
    """
    return patches_root() / AGGREGATE_PATCH_NAME
