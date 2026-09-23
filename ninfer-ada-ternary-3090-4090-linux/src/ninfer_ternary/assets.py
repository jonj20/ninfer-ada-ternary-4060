"""资产定位：同一套代码要同时服务仓库检出与已安装的 wheel。

仓库里打包器在 tools/、补丁在 patches/、引擎可执行文件在构建目录；安装之后这些内容被
塞进包内的 _data/ 下。把这个差异收敛到本模块，其余代码只问"路径在哪"，不问"我在哪种
形态里运行"。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 已安装形态下随包分发的数据目录名。
DATA_DIRNAME = "_data"

#: 指向 ninfer 源码树（提供 tools/artifact 制品读写）的环境变量名。
UPSTREAM_ENV = "NINFER_ROOT"

#: 覆盖引擎可执行文件目录的环境变量名。
ENGINE_BIN_ENV = "NINFER_ENGINE_BIN"

#: 引擎随包分发的可执行文件名。
ENGINE_PROGRAMS = ("ninfer", "ninfer-serve")


def package_dir() -> Path:
    """返回本包的安装目录。

    Returns:
        本包所在目录的绝对路径；安装形态下即 site-packages/ninfer_ternary。
    """
    return Path(__file__).resolve().parent


def bundled_data_dir() -> Path:
    """返回随包分发的数据目录。

    Returns:
        _data 目录路径；仓库形态下该目录不存在。
    """
    return package_dir() / DATA_DIRNAME


def is_bundled() -> bool:
    """判断当前是否运行在已安装形态。

    判据是随包数据目录里有没有打包器，而不是"路径看起来像不像 site-packages"。

    Returns:
        随包分发了打包器时为 True。
    """
    return (bundled_data_dir() / "pack").is_dir()


def repo_root() -> Path:
    """返回仓库检出根目录。

    Returns:
        仓库根目录路径；安装形态下该路径无意义，调用方应先用 is_bundled() 判断。
    """
    return package_dir().parents[1]


def patches_root() -> Path:
    """返回补丁数据目录（manifest.json 与 changed-files）。

    Returns:
        补丁目录路径。安装形态取随包副本，仓库形态取仓库里的 patches/。
    """
    bundled = bundled_data_dir() / "patches"
    return bundled if bundled.is_dir() else repo_root() / "patches"


def pack_dir() -> Path:
    """返回打包器脚本目录。

    Returns:
        含 pack.py 的目录路径。安装形态取随包副本，仓库形态取仓库里的 tools/。
    """
    bundled = bundled_data_dir() / "pack"
    return bundled if bundled.is_dir() else repo_root() / "tools"


def upstream_root() -> Path | None:
    """返回提供 tools/artifact 的目录。

    安装形态下上游源码树在编译完就被删掉了，随包分发的是制品的读写模块本身；仓库形态下
    它由环境变量指向真正 ninfer 检出。

    Returns:
        可用的上游根目录；仓库形态且未设置 NINFER_ROOT 时为 None。
    """
    bundled = bundled_data_dir() / "upstream"
    if (bundled / "tools" / "artifact").is_dir():
        return bundled
    raw = os.environ.get(UPSTREAM_ENV)
    if raw:
        candidate = Path(raw).expanduser()
        if (candidate / "tools" / "artifact").is_dir():
            return candidate.resolve()
    return None


def engine_binary(name: str) -> Path | None:
    """定位一个引擎可执行文件。

    Args:
        name: 可执行文件名，取 ENGINE_PROGRAMS 中的一项。

    Returns:
        存在的可执行文件路径；找不到时为 None。
    """
    candidates = [bundled_data_dir() / "bin" / name]
    override = os.environ.get(ENGINE_BIN_ENV)
    if override:
        candidates.append(Path(override).expanduser() / name)
    candidates.append(repo_root() / "build" / "apps" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
