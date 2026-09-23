"""公共引导：定位提供 tools/artifact 的根目录，并把相关目录挂到 sys.path。

打包器（pack.py）与 tools/verify/ 下的脚本需要制品读写模块 tools.artifact。
解析顺序收敛在本模块：

1. 环境变量 ``NINFER_ROOT`` —— 完整 ninfer 检出（引擎构建 / oracle 仍必填）；
2. 本仓自带的 ``tools/artifact`` 快照 —— 仅打包时足够，Linux/Windows 无需外置树。

只有打包/校验时，不设 NINFER_ROOT 也可以；编译引擎仍要把 NINFER_ROOT 指到检出。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: 指向 ninfer 源码树根目录的环境变量名。
NINFER_ROOT_ENV = "NINFER_ROOT"

#: 本仓根（tools/ 的父目录）；模块加载早期即需要，不依赖后续 def。
_REPO_ROOT = Path(__file__).resolve().parent.parent

_MISSING_HINT = (
    f"找不到 tools/artifact\n"
    f"  已查: 环境变量 {NINFER_ROOT_ENV}、本仓 {_REPO_ROOT}\n"
    f"  打包可使用仓内快照（tools/artifact）；编译引擎请设置 "
    f"{NINFER_ROOT_ENV}=<ninfer 检出根目录>。"
)


def tools_dir() -> Path:
    """返回本仓 tools/ 目录。

    Returns:
        本仓 tools/ 目录的绝对路径。
    """
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """返回本仓根目录。

    Returns:
        本仓根目录的绝对路径。
    """
    return tools_dir().parent


def bundled_artifact_root() -> Path | None:
    """若本仓自带可用的 tools/artifact，返回其父根（本仓根）。

    Returns:
        含 tools/artifact 的本仓根目录；快照不完整时为 None。
    """
    root = repo_root()
    artifact = root / "tools" / "artifact"
    if (artifact / "container.py").is_file() and (artifact / "numeric.py").is_file():
        return root
    return None


def ninfer_root() -> Path:
    """解析提供 tools/artifact 的根目录。

    Returns:
        ``NINFER_ROOT`` 指向的检出，或本仓自带快照的根。

    Raises:
        SystemExit: 环境变量无效且仓内无可用快照。
    """
    raw = os.environ.get(NINFER_ROOT_ENV)
    if raw:
        root = Path(raw).expanduser().resolve()
        if (root / "tools" / "artifact").is_dir():
            return root
        bundled = bundled_artifact_root()
        if bundled is not None:
            # 显式指错时仍继续，但环境变量语义是“引擎树”；打包用仓内快照。
            # 引擎构建脚本会单独校验 src/，此处不抢报错。
            pass
        else:
            raise SystemExit(
                f"{NINFER_ROOT_ENV}={root} 下找不到 tools/artifact，"
                "请指向 ninfer 源码树根目录，或使用仓内 tools/artifact 快照。"
            )
        if bundled is not None:
            return bundled
        raise SystemExit(
            f"{NINFER_ROOT_ENV}={root} 下找不到 tools/artifact，"
            "请指向 ninfer 源码树根目录。"
        )

    bundled = bundled_artifact_root()
    if bundled is not None:
        return bundled
    raise SystemExit(_MISSING_HINT)


def bootstrap() -> Path:
    """挂好搜索路径并返回提供 tools/artifact 的根目录。

    根目录最后被插入、因而排在最前，保证 ``tools`` 解析到该根下的包
    （本仓 tools/ 本身没有 __init__.py，靠命名空间包挂接）。

    Returns:
        含 tools/artifact 的根目录绝对路径。
    """
    root = ninfer_root()
    for entry in (repo_root(), tools_dir(), root):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root
