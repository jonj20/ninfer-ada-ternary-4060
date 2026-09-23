"""从 ninfer 检出重新生成补丁快照、逐文件摘要与聚合 diff。

补丁包的三个产物必须一起更新才可用：`changed-files/` 是 apply 真正落盘的内容，
`manifest.json` 是逐文件摘要（status / apply 靠它判断分叉），聚合 .patch 只供审阅。
只更新其中一个会让三者互相矛盾，而矛盾要等到别人 apply 时才暴露出来。
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import ManifestError, PatchManifest
from .patchset import sha256_file
from .scratch import temp_root

_LOGGER = logging.getLogger("ninfer_ternary")


class ExportError(RuntimeError):
    """快照导出失败。"""


@dataclass(frozen=True, slots=True)
class ExportResult:
    """一次导出的结果。

    Attributes:
        written: 写入快照的文件数。
        digest_changed: 摘要相对清单发生变化的文件路径。
        appended: 本次新纳入清单的文件路径。
        unmanaged: 检出里有改动、却不在清单里的文件路径。
    """

    written: int
    digest_changed: tuple[str, ...]
    appended: tuple[str, ...] = field(default=())
    unmanaged: tuple[str, ...] = field(default=())


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    """执行一条外部命令并返回结果，不因非零退出码抛异常。

    Args:
        command: 完整命令及其参数。

    Returns:
        子进程结果。
    """
    return subprocess.run(command, capture_output=True, check=False)


def _git(repo: Path, *args: str) -> str:
    """在检出里执行一次 git 并返回标准输出。

    Args:
        repo: ninfer 源码树根目录。
        args: 传给 git 的参数。

    Returns:
        标准输出文本。

    Raises:
        ExportError: git 以非零状态退出。
    """
    completed = _run(["git", "-C", str(repo), *args])
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise ExportError(f"git {' '.join(args)} 失败: {detail}")
    return completed.stdout.decode("utf-8", "replace")


def _upstream_bytes(repo: Path, commit: str, path: str) -> bytes | None:
    """取出某文件在目标提交上的原始内容。

    Args:
        repo: ninfer 源码树根目录。
        commit: 目标提交。
        path: 相对检出根目录的路径。

    Returns:
        原始内容；该路径在目标提交上不存在时返回 None。
    """
    completed = _run(["git", "-C", str(repo), "show", f"{commit}:{path}"])
    if completed.returncode != 0:
        return None
    return completed.stdout


def _unified_diff(path: str, upstream: bytes | None, current: Path) -> str:
    """生成单个文件的统一 diff，头两行固定为 a/<path> 与 b/<path>。

    两侧都显式给标签，而不是让 diff 打出临时文件的路径 —— 聚合补丁是给人读的，
    它必须能直接喂给 patch -p1。

    Args:
        path: 相对检出根目录的路径。
        upstream: 上游内容；新增文件为 None。
        current: 检出中的当前文件路径。

    Returns:
        统一 diff 文本。

    Raises:
        ExportError: diff 以 0/1 之外的状态退出。
    """
    with tempfile.TemporaryDirectory(prefix="export-", dir=temp_root()) as tmp:
        left = Path(tmp) / "upstream"
        left.write_bytes(upstream if upstream is not None else b"")
        completed = _run(
            [
                "diff",
                "-u",
                f"--label=a/{path}",
                f"--label=b/{path}",
                str(left),
                str(current),
            ]
        )
        if completed.returncode not in (0, 1):
            detail = completed.stderr.decode("utf-8", "replace").strip()
            raise ExportError(f"diff {path} 失败: {detail}")
        return completed.stdout.decode("utf-8", "replace")


def _unmanaged_paths(repo: Path, managed: set[str]) -> tuple[str, ...]:
    """列出检出里被改动、却不在清单里的文件。

    这份列表刻意只报告、不处理：清单是人工筛定的范围，多出来的改动多半是本地调试残留
    （例如为让上游测试编过而补的 include），不该被顺手吸进补丁包。要纳入就得显式 --add。

    Args:
        repo: ninfer 源码树根目录。
        managed: 清单覆盖的路径集合。

    Returns:
        未纳入清单的改动路径，已排序。
    """
    found: list[str] = []
    for line in _git(repo, "status", "--porcelain", "-uall").splitlines():
        if len(line) < 4:
            continue
        relative = line[3:].strip()
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        if relative and relative not in managed:
            found.append(relative)
    return tuple(sorted(found))


def _write_document(path: Path, document: dict[str, Any]) -> None:
    """把清单文档写回磁盘，保持 2 空格缩进与尾换行。

    Args:
        path: manifest.json 的路径。
        document: 清单的 JSON 文档。
    """
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _entry_document(repo: Path, commit: str, path: str) -> dict[str, Any]:
    """为新纳入清单的路径构造一个条目。

    条目形状与既有条目逐键一致，避免同一份清单里出现两种写法的条目。

    Args:
        repo: ninfer 源码树根目录。
        commit: 目标提交。
        path: 相对检出根目录的路径。

    Returns:
        清单条目文档。

    Raises:
        ExportError: 检出不包含该文件。
    """
    source = repo / path
    if not source.is_file():
        raise ExportError(f"{path}: 检出里找不到该文件")
    digest = sha256_file(source)
    upstream = _upstream_bytes(repo, commit, path)
    if upstream is None:
        return {"path": path, "sha256": digest, "status": "added"}
    return {
        "path": path,
        "sha256": digest,
        "status": "modified",
        "sha256_upstream": hashlib.sha256(upstream).hexdigest(),
    }


def export_snapshot(
    repo: Path,
    manifest_path: Path,
    snapshot_root: Path,
    patch_path: Path,
    add: Sequence[str] = (),
) -> ExportResult:
    """用检出的当前内容刷新补丁快照、清单摘要与聚合 diff。

    清单的范围是人工筛定的：只有显式写进 `add` 的路径才会被纳入，检出里的其它改动一律
    只报告。这样"改了引擎却忘了更新补丁包"会以"摘要变化"暴露，而本地调试残留不会。

    Args:
        repo: ninfer 源码树根目录（已应用本补丁的工作副本）。
        manifest_path: patches/manifest.json 的路径。
        snapshot_root: patches/changed-files 目录。
        patch_path: 聚合 diff 的输出路径。
        add: 需要新纳入清单的路径（相对检出根目录）。

    Returns:
        导出结果。

    Raises:
        ManifestError: 清单结构不符，或清单里的文件在检出中不存在。
        ExportError: git 或 diff 调用失败，或修改条目在目标提交上找不到对应路径。
    """
    initial = PatchManifest.load(manifest_path)
    document: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, Any]] = {item["path"]: item for item in document["files"]}

    appended: list[str] = []
    for path in add:
        if path in entries:
            continue
        item = _entry_document(repo, initial.target_commit, path)
        document["files"].append(item)
        entries[path] = item
        appended.append(path)
    if appended:
        document["files"].sort(key=lambda item: item["path"])
        document["file_count"] = len(document["files"])
        document["added_count"] = sum(1 for i in document["files"] if i["status"] == "added")
        document["modified_count"] = document["file_count"] - document["added_count"]
        _write_document(manifest_path, document)

    manifest = PatchManifest.load(manifest_path)
    changed: list[str] = []
    chunks: list[str] = []
    for entry in manifest.files:
        source = repo / entry.path
        if not source.is_file():
            raise ManifestError(f"{entry.path}: 检出里找不到该文件")
        target = snapshot_root / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

        digest = sha256_file(source)
        if digest != entry.sha256:
            changed.append(entry.path)
        entries[entry.path]["sha256"] = digest

        upstream: bytes | None = None
        if entry.status != "added":
            upstream = _upstream_bytes(repo, manifest.target_commit, entry.path)
            if upstream is None:
                raise ExportError(
                    f"{entry.path}: 目标提交 {manifest.target_commit} 上没有该路径，"
                    "清单里的 status 可能标错了"
                )
        chunks.append(_unified_diff(entry.path, upstream, source))

    _write_document(manifest_path, document)
    patch_path.write_text("".join(chunks), encoding="utf-8")

    unmanaged = _unmanaged_paths(repo, {entry.path for entry in manifest.files})
    _LOGGER.info(
        "导出完成: files=%d appended=%d changed=%d",
        len(manifest.files),
        len(appended),
        len(changed),
    )
    return ExportResult(
        written=len(manifest.files),
        digest_changed=tuple(changed),
        appended=tuple(appended),
        unmanaged=unmanaged,
    )
