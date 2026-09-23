"""PEP 517 构建后端：让 uv tool install 一条命令就拿到编译好的引擎。

源码与三元改动已合入本仓：build_wheel 默认**树内 cmake**（仓根 → 临时构建目录 →
可执行文件 + tools/artifact + pack 脚本打进 wheel），不再 clone/打 patches。

环境变量：
    NINFER_TERNARY_SKIP_BUILD=1  只装 Python 侧，不编译，wheel 退化为纯 Python
    NINFER_TERNARY_KEEP_BUILD=1  保留临时构建目录，编译失败时排查用
    NINFER_TERNARY_SOURCE        覆盖源码树（默认本仓根）
    NINFER_TERNARY_TMPDIR        改写临时根目录，默认 $TMPDIR/ninfer-ternary
    其余见 ninfer_ternary.engine 的模块说明。
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import logging
import os
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parent
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ninfer_ternary.engine import BuildRequest, build_engine  # noqa: E402
from ninfer_ternary.manifest import PatchManifest, default_manifest_path  # noqa: E402
from ninfer_ternary.scratch import prune_temp_root, temp_root  # noqa: E402

_LOGGER = logging.getLogger("ninfer_ternary.build_backend")
_GENERATOR = "ninfer-ternary build_backend"
_MODULE = "ninfer_ternary"
_SKIP_BUILD_ENV = "NINFER_TERNARY_SKIP_BUILD"
_TRUE_VALUES = ("1", "true", "yes", "on")
_SCRATCH_PREFIX = "wheel-"
_WHEEL_TAG_BINARY = "py3-none-linux_x86_64"
_WHEEL_TAG_PURE = "py3-none-any"
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_SDIST_ROOT_FILES = (
    "pyproject.toml",
    "build_backend.py",
    "justfile",
    "README.md",
    "LICENSE",
    "NOTICE",
    "uv.lock",
    "manifest.json",
    "CMakeLists.txt",
)
# patches/ 已随源码合入删除；列表里保留名字也不会炸（_sdist_paths 跳过缺失目录）。
_SDIST_ROOT_DIRS = ("src", "tools", "tests", "docs", "include", "apps", "cmake", "third_party")
_SDIST_EXCLUDES = frozenset(
    {
        ".git",
        ".venv",
        ".agents",
        "out",
        "dist",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
)
#: 随包分发的打包器文件（相对仓库 tools/）。
_PACK_FILES = ("pack.py", "_bootstrap.py", "_ternary_ref.py", "MAPPING.json")


def _configure_logging() -> None:
    """把构建进度打到 stderr，供前端原样透传。"""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(levelname)s %(message)s",
    )


def _flag(name: str) -> bool:
    """读取一个布尔型环境变量。

    Args:
        name: 环境变量名。

    Returns:
        变量值为 1 / true / yes / on（大小写不敏感）时为 True。
    """
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _project() -> dict[str, Any]:
    """读取 pyproject.toml 的 project 段。

    Returns:
        project 段的内容。

    Raises:
        RuntimeError: 缺少 project 段。
    """
    with (ROOT / "pyproject.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    project = raw.get("project")
    if not isinstance(project, dict):
        raise RuntimeError("pyproject.toml 缺少 project 段")
    return project


def _short_name(project: dict[str, Any]) -> str:
    """返回 wheel 文件名里使用的发行名。

    Args:
        project: pyproject.toml 的 project 段。

    Returns:
        下划线形式的发行名。
    """
    return str(project["name"]).replace("-", "_")


def _dist_info(project: dict[str, Any]) -> str:
    """返回 dist-info 目录名。

    Args:
        project: pyproject.toml 的 project 段。

    Returns:
        <发行名>-<版本>.dist-info。
    """
    return f"{_short_name(project)}-{project['version']}.dist-info"


def _iter_tree(root: Path) -> Iterator[Path]:
    """列出目录下的全部普通文件，跳过字节码缓存。

    Args:
        root: 根目录；不存在时产出为空。

    Yields:
        目录中的文件路径。
    """
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            yield path


def _metadata(project: dict[str, Any]) -> str:
    """拼出 METADATA。

    Args:
        project: pyproject.toml 的 project 段。

    Returns:
        core metadata 文本。
    """
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        "License-Expression: Apache-2.0",
    ]
    for path in (ROOT / "LICENSE", ROOT / "NOTICE"):
        if path.is_file():
            lines.append(f"License-File: {path.name}")
    for require in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {require}")
    if project.get("requires-python"):
        lines.append(f"Requires-Python: {project['requires-python']}")
    lines.append("Classifier: License :: OSI Approved :: Apache Software License")
    lines.append("Classifier: Programming Language :: Python :: 3.12")
    lines.append("Classifier: Topic :: Scientific/Engineering :: Artificial Intelligence")
    readme = ROOT / str(project.get("readme", "README.md"))
    if readme.is_file():
        lines.append("Description-Content-Type: text/markdown")
        lines.append("")
        lines.append(readme.read_text(encoding="utf-8"))
    return "\n".join(lines) + "\n"


def _wheel_metadata(*, pure_python: bool) -> str:
    """拼出 WHEEL。

    Args:
        pure_python: 是否不含平台相关文件。

    Returns:
        WHEEL 文本。
    """
    tag = _WHEEL_TAG_PURE if pure_python else _WHEEL_TAG_BINARY
    return (
        "Wheel-Version: 1.0\n"
        f"Generator: {_GENERATOR}\n"
        f"Root-Is-Purelib: {'true' if pure_python else 'false'}\n"
        f"Tag: {tag}\n"
    )


def _entry_points(project: dict[str, Any]) -> str:
    """按 pyproject 的 scripts 段生成 entry_points.txt。

    Args:
        project: pyproject.toml 的 project 段。

    Returns:
        entry_points.txt 文本；没有脚本时为空串。
    """
    scripts = project.get("scripts") or {}
    if not scripts:
        return ""
    body = "\n".join(f"{name} = {target}" for name, target in scripts.items())
    return f"[console_scripts]\n{body}\n"


def _digest(payload: bytes) -> tuple[str, int]:
    """计算 wheel RECORD 需要的摘要与长度。

    Args:
        payload: 文件内容。

    Returns:
        （无填充的 urlsafe base64 sha256, 字节数）。
    """
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
    return digest, len(payload)


def _write_member(
    archive: zipfile.ZipFile,
    arcname: str,
    payload: bytes,
    *,
    executable: bool = False,
) -> None:
    """往 wheel 里写一个成员，日期固定以便复现。

    Args:
        archive: 目标压缩包。
        arcname: 包内路径。
        payload: 文件内容。
        executable: 是否打上可执行权限位。
    """
    info = zipfile.ZipInfo(arcname, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o755 if executable else 0o644) << 16
    archive.writestr(info, payload)


def _collect(
    project: dict[str, Any],
    binaries: dict[str, Path],
    upstream: Path | None,
) -> list[tuple[str, bytes, bool]]:
    """列出 wheel 的全部成员。

    Args:
        project: pyproject.toml 的 project 段。
        binaries: 可执行文件名到构建产物的映射。
        upstream: 上游制品读写模块的落点目录；未编译时为 None。

    Returns:
        （包内路径, 内容, 是否可执行）三元组列表。
    """
    members: list[tuple[str, bytes, bool]] = []
    module_root = _SRC / _MODULE
    for path in _iter_tree(module_root):
        arcname = f"{_MODULE}/{path.relative_to(module_root).as_posix()}"
        members.append((arcname, path.read_bytes(), False))
    # patches/ 已删除；若将来恢复快照目录仍打进 wheel。
    for path in _iter_tree(ROOT / "patches"):
        arcname = f"{_MODULE}/_data/patches/{path.relative_to(ROOT / 'patches').as_posix()}"
        members.append((arcname, path.read_bytes(), False))
    for name in _PACK_FILES:
        source = ROOT / "tools" / name
        if source.is_file():
            members.append((f"{_MODULE}/_data/pack/{name}", source.read_bytes(), False))
    if upstream is not None:
        for path in _iter_tree(upstream):
            arcname = f"{_MODULE}/_data/upstream/{path.relative_to(upstream).as_posix()}"
            members.append((arcname, path.read_bytes(), False))
    for name, path in sorted(binaries.items()):
        members.append((f"{_MODULE}/_data/bin/{name}", path.read_bytes(), True))
    members.extend(_dist_info_members(project, pure_python=not binaries))
    return members


def _dist_info_members(
    project: dict[str, Any],
    *,
    pure_python: bool,
) -> list[tuple[str, bytes, bool]]:
    """列出 dist-info 的成员。

    Args:
        project: pyproject.toml 的 project 段。
        pure_python: 该 wheel 是否不含平台相关文件。

    Returns:
        （包内路径, 内容, 是否可执行）三元组列表。
    """
    info = _dist_info(project)
    members = [
        (f"{info}/METADATA", _metadata(project).encode("utf-8"), False),
        (f"{info}/WHEEL", _wheel_metadata(pure_python=pure_python).encode("utf-8"), False),
    ]
    entry_points = _entry_points(project)
    if entry_points:
        members.append((f"{info}/entry_points.txt", entry_points.encode("utf-8"), False))
    for name in ("LICENSE", "NOTICE"):
        source = ROOT / name
        if source.is_file():
            members.append((f"{info}/licenses/{name}", source.read_bytes(), False))
    return members


def _record(rows: Sequence[tuple[str, str, int]]) -> bytes:
    """拼出 RECORD。

    Args:
        rows: （包内路径, 摘要, 长度）列表。

    Returns:
        RECORD 文本的字节。
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for arcname, digest, size in rows:
        writer.writerow([arcname, f"sha256={digest}", size])
    return buffer.getvalue().encode("utf-8")


def _write_wheel(
    path: Path,
    info: str,
    members: Sequence[tuple[str, bytes, bool]],
) -> str:
    """按 PEP 427 写出一个 wheel，并附上 RECORD。

    Args:
        path: 输出文件路径。
        info: dist-info 目录名。
        members: （包内路径, 内容, 是否可执行）列表。

    Returns:
        生成的文件名。
    """
    rows: list[tuple[str, str, int]] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, payload, executable in members:
            _write_member(archive, arcname, payload, executable=executable)
            digest, size = _digest(payload)
            rows.append((arcname, digest, size))
        record_name = f"{info}/RECORD"
        rows.append((record_name, "", 0))
        _write_member(archive, record_name, _record(rows))
    return path.name


def _assemble(
    wheel_directory: Path,
    project: dict[str, Any],
    binaries: dict[str, Path],
    upstream: Path | None,
) -> str:
    """把成员写成一个 wheel。

    Args:
        wheel_directory: wheel 的输出目录。
        project: pyproject.toml 的 project 段。
        binaries: 可执行文件名到构建产物的映射。
        upstream: 上游制品读写模块的落点目录；未编译时为 None。

    Returns:
        生成的文件名。
    """
    tag = _WHEEL_TAG_BINARY if binaries else _WHEEL_TAG_PURE
    filename = f"{_short_name(project)}-{project['version']}-{tag}.whl"
    return _write_wheel(
        wheel_directory / filename,
        _dist_info(project),
        _collect(project, binaries, upstream),
    )


def _copy_artifact_module(source_root: Path, destination: Path) -> Path:
    """把仓内 tools/artifact 拷到 destination（调用方传入 staging/upstream）。

    Args:
        source_root: 含 tools/artifact 的源码树根。
        destination: 落点；须为 .../upstream，其下生成 tools/artifact。

    Returns:
        destination 本身。

    Raises:
        RuntimeError: 源码树缺少 tools/artifact。
    """
    import shutil

    origin = source_root / "tools" / "artifact"
    if not origin.is_dir():
        raise RuntimeError(f"源码树缺少 tools/artifact: {origin}")
    if destination.name != "upstream":
        destination = destination / "upstream"
    package = destination / "tools"
    if package.exists():
        shutil.rmtree(package)
    shutil.copytree(origin, package / "artifact", ignore=shutil.ignore_patterns("__pycache__"))
    (package / "__init__.py").write_text(
        '"""随 wheel 分发的制品读写模块（来自本仓 tools/artifact）。"""\n', encoding="utf-8"
    )
    return destination


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """构建 wheel；默认树内 cmake 编译引擎并打进包。

    Args:
        wheel_directory: wheel 的输出目录。
        config_settings: 前端传入的配置；本后端不使用。
        metadata_directory: 已生成的 dist-info 目录；本后端不使用。

    Returns:
        生成的文件名。

    Raises:
        EngineError: 引擎构建失败。
    """
    _configure_logging()
    project = _project()
    try:
        with tempfile.TemporaryDirectory(prefix=_SCRATCH_PREFIX, dir=temp_root()) as staging_raw:
            staging = Path(staging_raw)
            binaries: dict[str, Path] = {}
            upstream: Path | None = None
            if _flag(_SKIP_BUILD_ENV):
                _LOGGER.warning("跳过引擎编译: %s=1", _SKIP_BUILD_ENV)
                # 纯 Python wheel 仍带上仓内 tools/artifact，保证 ninfer-convert 可用。
                upstream = staging / "upstream"
                _copy_artifact_module(ROOT, upstream)
            else:
                manifest = PatchManifest.load(default_manifest_path())
                request = BuildRequest.from_environment(manifest)
                # 树内一体：源码默认本仓根；from_environment 已处理。
                result = build_engine(
                    staging / "bin",
                    request,
                    assets_dir=staging,
                )
                binaries = dict(result.programs)
                upstream = staging / "upstream"
                _LOGGER.info("落地自检: %d/%d", result.self_check_passed, result.self_check_total)
            return _assemble(Path(wheel_directory), project, binaries, upstream)
    finally:
        prune_temp_root()


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    """只生成 dist-info，不触发编译。

    Args:
        metadata_directory: dist-info 的父目录。
        config_settings: 前端传入的配置；本后端不使用。

    Returns:
        dist-info 目录名。
    """
    project = _project()
    target = Path(metadata_directory) / _dist_info(project)
    (target / "licenses").mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_text(_metadata(project), encoding="utf-8")
    (target / "WHEEL").write_text(_wheel_metadata(pure_python=False), encoding="utf-8")
    entry_points = _entry_points(project)
    if entry_points:
        (target / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    for name in ("LICENSE", "NOTICE"):
        source = ROOT / name
        if source.is_file():
            (target / "licenses" / name).write_bytes(source.read_bytes())
    return target.name


def _sdist_paths() -> Iterator[Path]:
    """列出 sdist 需要收录的文件。

    Yields:
        仓库内需要打包的文件路径。
    """
    for name in _SDIST_ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            yield path
    for name in _SDIST_ROOT_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not (_SDIST_EXCLUDES & set(path.parts)):
                yield path


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """构建可编辑安装用的 wheel。

    开发（uv run / uv sync）走这条路径：只写一个指向 src/ 的 .pth，不编译引擎 —— 否则
    每次跑 lint 都要等一遍 CUDA 编译。

    Args:
        wheel_directory: wheel 的输出目录。
        config_settings: 前端传入的配置；本后端不使用。
        metadata_directory: 已生成的 dist-info 目录；本后端不使用。

    Returns:
        生成的文件名。
    """
    project = _project()
    dist = _short_name(project)
    filename = f"{dist}-{project['version']}-py3-none-any.whl"
    pth = f"__editable__.{dist}-{project['version']}.pth"
    members: list[tuple[str, bytes, bool]] = [(pth, (str(_SRC) + "\n").encode("utf-8"), False)]
    members.extend(_dist_info_members(project, pure_python=True))
    return _write_wheel(Path(wheel_directory) / filename, _dist_info(project), members)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    """构建源码包，供离线环境或二次打包使用。

    Args:
        sdist_directory: sdist 的输出目录。
        config_settings: 前端传入的配置；本后端不使用。

    Returns:
        生成的文件名。
    """
    project = _project()
    base = f"{_short_name(project)}-{project['version']}"
    target = Path(sdist_directory) / f"{base}.tar.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        for path in _sdist_paths():
            arcname = f"{base}/{path.relative_to(ROOT).as_posix()}"
            info = archive.gettarinfo(str(path), arcname=arcname)
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return target.name


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    """声明构建 wheel 的额外依赖。

    Args:
        config_settings: 前端传入的配置；本后端不使用。

    Returns:
        空列表：后端只用标准库，编译依赖由引擎自检报错。
    """
    return []


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    """声明构建可编辑安装的额外依赖。

    Args:
        config_settings: 前端传入的配置；本后端不使用。

    Returns:
        空列表。
    """
    return []


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    """声明构建 sdist 的额外依赖。

    Args:
        config_settings: 前端传入的配置；本后端不使用。

    Returns:
        空列表。
    """
    return []
