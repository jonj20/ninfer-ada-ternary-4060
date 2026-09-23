"""拉取上游 ninfer、打上三元补丁、自检、编译，并把临时树清干净。

这条流程有两个调用方：wheel 构建后端（uv tool install 时自动跑）与仓库内的
build-engine 子命令。两者的纪律一致 —— 克隆出来的源码树、CMake 构建目录、编译子进程的
临时文件全部落在 $TMPDIR/ninfer-ternary 这一个根下，用完即删；只有最终的可执行文件被拷走。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .assets import ENGINE_PROGRAMS
from .checks import check_all
from .manifest import PatchManifest, changed_files_root, default_manifest_path
from .patchset import apply_patch_set
from .scratch import prune_temp_root, temp_root

_LOGGER = logging.getLogger(__name__)

#: 覆盖上游仓库地址（离线镜像、本地路径、内网 fork）。
TARGET_REPO_ENV = "NINFER_TERNARY_TARGET_REPO"
#: 直接复用一棵已有的目标检出，跳过克隆。
SOURCE_ENV = "NINFER_TERNARY_SOURCE"
#: 临时树的父目录；默认是 $TMPDIR/ninfer-ternary。
SCRATCH_ENV = "NINFER_TERNARY_BUILD_ROOT"
#: 置 1 时保留临时树，便于排查编译失败。
KEEP_ENV = "NINFER_TERNARY_KEEP_BUILD"
#: CUDA 架构，86 或 89。
ARCH_ENV = "NINFER_TERNARY_ARCH"
#: 并行度。
JOBS_ENV = "NINFER_TERNARY_JOBS"
#: 置 1 时连引擎自带测试套件一起编译并跑一遍。
TESTS_ENV = "NINFER_TERNARY_RUN_TESTS"
#: 服务端内嵌 Web UI 的开关；置 0 可跳过 3 MiB 的 GitHub 发布包下载。
UI_ENV = "NINFER_TERNARY_ENABLE_UI"

#: 临时树的名字前缀。所有临时物都在 ninfer-ternary/ 根下，前缀只区分用途。
_SCRATCH_PREFIX = "build-"

#: 允许的 CUDA 架构。上游用 FATAL_ERROR 硬拒其它架构，这里提前拦下更省事。
_SUPPORTED_ARCHES = ("86", "89")

_TRUE_VALUES = ("1", "true", "yes", "on")


class EngineError(RuntimeError):
    """引擎构建无法完成。"""


def _flag(name: str) -> bool:
    """读取一个布尔型环境变量。

    Args:
        name: 环境变量名。

    Returns:
        变量值为 1 / true / yes / on（大小写不敏感）时为 True。
    """
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _positive_int(name: str, default: int) -> int:
    """读取一个正整数型环境变量。

    Args:
        name: 环境变量名。
        default: 变量缺省或非法时的回退值。

    Returns:
        解析出的正整数。
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        _LOGGER.warning("忽略非整数环境变量: %s=%s", name, raw)
        return default
    return value if value > 0 else default


def _require_tool(name: str) -> None:
    """确认构建依赖在 PATH 上。

    Args:
        name: 可执行文件名。

    Raises:
        EngineError: 找不到该命令。
    """
    if shutil.which(name) is None:
        raise EngineError(f"缺少构建依赖: {name}（见 README 的依赖安装一节）")


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """执行外部命令，失败即抛。

    Args:
        command: 命令与参数。
        cwd: 工作目录；None 表示继承当前目录。
        env: 子进程环境；None 表示继承当前环境。

    Raises:
        EngineError: 命令返回非零。
    """
    _LOGGER.info("执行: %s", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        raise EngineError(f"命令失败（exit {completed.returncode}）: {' '.join(command)}")


@dataclass(frozen=True, slots=True)
class BuildRequest:
    """一次引擎构建的全部输入。

    Attributes:
        repository: 上游仓库地址或本地路径。
        commit: 要检出的提交。
        arch: CUDA 架构，86 或 89。
        jobs: 编译并行度。
        source: 已有的目标检出；给定时跳过克隆，且不会被删除。
        scratch: 临时树的父目录；None 表示 $TMPDIR/ninfer-ternary。
        keep: 是否保留临时树。
        run_tests: 是否编译并运行引擎自带测试套件。
    """

    repository: str
    commit: str
    arch: str = "89"
    jobs: int = 8
    source: Path | None = None
    scratch: Path | None = None
    keep: bool = False
    run_tests: bool = False

    @classmethod
    def from_environment(cls, manifest: PatchManifest) -> "BuildRequest":
        """按环境变量构造请求。

        Args:
            manifest: 补丁清单，提供默认的仓库地址与提交。

        Returns:
            构造好的请求。

        Raises:
            EngineError: CUDA 架构不在支持范围内。
        """
        arch = os.environ.get(ARCH_ENV, "89").strip() or "89"
        if arch not in _SUPPORTED_ARCHES:
            raise EngineError(f"{ARCH_ENV} 只能是 86 或 89，实际为 {arch}")
        source = os.environ.get(SOURCE_ENV, "").strip()
        scratch = os.environ.get(SCRATCH_ENV, "").strip()
        return cls(
            repository=os.environ.get(TARGET_REPO_ENV, "").strip() or manifest.target_repository,
            commit=manifest.target_commit,
            arch=arch,
            jobs=_positive_int(JOBS_ENV, os.cpu_count() or 8),
            source=Path(source).expanduser().resolve() if source else None,
            scratch=Path(scratch).expanduser().resolve() if scratch else None,
            keep=_flag(KEEP_ENV),
            run_tests=_flag(TESTS_ENV),
        )


@dataclass(frozen=True, slots=True)
class BuildResult:
    """一次引擎构建的产出。

    Attributes:
        programs: 可执行文件名到目标路径的映射，已按名字拷贝到位。
        scratch: 本次使用的临时目录（keep 为真时它仍然存在）。
        self_check_passed: 通过的自检条数。
        self_check_total: 自检总条数。
    """

    programs: Mapping[str, Path]
    scratch: Path
    self_check_passed: int
    self_check_total: int


def _fetch(repository: str, commit: str, dest: Path) -> None:
    """把指定提交取到目标目录。

    先试单提交浅取，失败再退回完整克隆 —— 前者快得多，但上游不一定允许按 sha 取。

    Args:
        repository: 仓库地址或本地路径。
        commit: 目标提交。
        dest: 目标目录，允许不存在。

    Raises:
        EngineError: 两种方式都失败，或取到的提交不是目标提交。
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        _run(["git", "init", "--quiet", str(dest)])
        _run(["git", "-C", str(dest), "remote", "add", "origin", repository])
        _run(["git", "-C", str(dest), "fetch", "--quiet", "--depth", "1", "origin", commit])
        _run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    except EngineError:
        _LOGGER.info("浅取失败，改用完整克隆: %s", repository)
        shutil.rmtree(dest, ignore_errors=True)
        _run(["git", "clone", "--quiet", repository, str(dest)])
        _run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", commit])


def _land(source: Path) -> tuple[int, int]:
    """把补丁落到源码树并做落地自检。

    Args:
        source: 目标源码树。

    Returns:
        （通过条数, 总条数）。

    Raises:
        EngineError: 自检未全通过。
    """
    manifest = PatchManifest.load(default_manifest_path())
    report = apply_patch_set(source, manifest, changed_files_root(), force=True)
    _LOGGER.info("落地补丁: files=%d", len(report.results))
    findings = check_all(source)
    passed = sum(1 for item in findings if item.ok)
    if passed != len(findings):
        broken = ", ".join(item.path for item in findings if not item.ok)
        raise EngineError(f"落地自检未通过（{passed}/{len(findings)}）: {broken}")
    return passed, len(findings)


def _compile_scratch(build_root: Path) -> Path:
    """返回编译子进程使用的 TMPDIR，落在临时树内部。

    Args:
        build_root: CMake 构建目录。

    Returns:
        已创建的目录路径。

    Raises:
        OSError: 目录无法创建。
    """
    path = build_root.parent / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compile(source: Path, build_root: Path, request: BuildRequest) -> dict[str, Path]:
    """配置并编译引擎可执行文件。

    Args:
        source: 已打好补丁的源码树。
        build_root: CMake 构建目录。
        request: 构建请求。

    Returns:
        可执行文件名到其构建产物的映射。

    Raises:
        EngineError: 配置、编译或测试失败。
    """
    _require_tool("cmake")
    _require_tool("ninja")
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build_root),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={request.arch}",
        "-DNINFER_BUILD_APPS=ON",
    ]
    if request.run_tests:
        configure.append("-DBUILD_TESTING=ON")
    ui = os.environ.get(UI_ENV, "").strip()
    if ui:
        configure.append(f"-DNINFER_ENABLE_UI={1 if ui.lower() in _TRUE_VALUES else 0}")
    # nvcc 把 tmpxft_* 中间文件丢在 TMPDIR 根下且名字里没有项目标识，只有把子进程的
    # TMPDIR 指进临时树，清理才是"删一个目录"而不是"认路径猜归属"。
    child_env = dict(os.environ, TMPDIR=str(_compile_scratch(build_root)))
    _run(configure, env=child_env)
    _run(
        [
            "cmake",
            "--build",
            str(build_root),
            "-j",
            str(request.jobs),
            "--target",
            *ENGINE_PROGRAMS,
        ],
        env=child_env,
    )
    if request.run_tests:
        _run(
            ["ctest", "--output-on-failure", "-j", str(request.jobs)],
            cwd=build_root,
            env=child_env,
        )
    return {name: build_root / "apps" / name for name in ENGINE_PROGRAMS}


def _copy_programs(programs: Mapping[str, Path], destination: Path) -> dict[str, Path]:
    """把编译产物拷到目标目录。

    Args:
        programs: 可执行文件名到构建产物的映射。
        destination: 目标目录，不存在时创建。

    Returns:
        可执行文件名到目标路径的映射。

    Raises:
        EngineError: 构建产物缺失。
    """
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for name, origin in programs.items():
        if not origin.is_file():
            raise EngineError(f"构建产物缺失: {origin}")
        target = destination / name
        shutil.copy2(origin, target)
        target.chmod(0o755)
        copied[name] = target
    return copied


def _collect_upstream(source: Path, destination: Path) -> Path:
    """把上游的制品读写模块拷出来，供安装形态在源码树被删掉之后继续用。

    上游的 tools/ 是没有 __init__.py 的命名空间包，装进 wheel 后必须补一个，否则
    tools.artifact 在 site-packages 里解析不到。

    Args:
        source: 已打好补丁的源码树。
        destination: 随包数据目录。

    Returns:
        落点目录（destination/upstream）。

    Raises:
        EngineError: 上游缺少 tools/artifact。
    """
    origin = source / "tools" / "artifact"
    if not origin.is_dir():
        raise EngineError(f"上游源码树缺少制品模块: {origin}")
    target = destination / "upstream"
    package = target / "tools"
    shutil.copytree(origin, package / "artifact", ignore=shutil.ignore_patterns("__pycache__"))
    (package / "__init__.py").write_text(
        '"""从上游 ninfer 检出取出的制品读写模块。"""\n', encoding="utf-8"
    )
    return target


def scratch_parent(request: BuildRequest) -> Path:
    """返回本次构建的临时树父目录，不存在时创建。

    Args:
        request: 构建请求；其 scratch 为 None 时落到项目临时根。

    Returns:
        已存在的父目录路径。

    Raises:
        OSError: 目录无法创建。
    """
    parent = request.scratch if request.scratch is not None else temp_root()
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def build_engine(
    destination: Path,
    request: BuildRequest,
    *,
    assets_dir: Path | None = None,
) -> BuildResult:
    """完整走一遍拉取、打补丁、自检、编译，并把可执行文件交到目标目录。

    Args:
        destination: 可执行文件的落点目录。
        request: 构建请求。
        assets_dir: 需要一并收集上游 Python 模块时的落点目录。

    Returns:
        构建结果。

    Raises:
        EngineError: 任一阶段失败；临时目录仍会按 request.keep 处理。
    """
    scratch = Path(tempfile.mkdtemp(prefix=_SCRATCH_PREFIX, dir=scratch_parent(request)))
    _LOGGER.info("临时目录: %s", scratch)
    try:
        if request.source is not None:
            source = request.source
            _LOGGER.info("复用已有检出: %s", source)
        else:
            source = scratch / "source"
            _LOGGER.info("拉取: %s@%s", request.repository, request.commit)
            _fetch(request.repository, request.commit, source)
        passed, total = _land(source)
        programs = _compile(source, scratch / "build", request)
        if assets_dir is not None:
            _collect_upstream(source, assets_dir)
        copied = _copy_programs(programs, destination)
        return BuildResult(
            programs=copied,
            scratch=scratch,
            self_check_passed=passed,
            self_check_total=total,
        )
    finally:
        if request.keep:
            _LOGGER.warning("按 %s 保留临时目录: %s", KEEP_ENV, scratch)
        else:
            shutil.rmtree(scratch, ignore_errors=True)
            prune_temp_root()
