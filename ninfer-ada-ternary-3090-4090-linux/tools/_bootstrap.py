"""公共引导：定位 ninfer 源码树，并把相关目录挂到 sys.path 上。

pack.py 与 tools/verify/ 下的脚本都要用到 ninfer 源码树自带的 tools/artifact（制品读写）。
本模块把"树在哪里"这一件事收敛到唯一一处，避免每个脚本各写一份绝对路径。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: 指向 ninfer 源码树根目录的环境变量名。
NINFER_ROOT_ENV = "NINFER_ROOT"

_MISSING_HINT = (
    "未设置 ninfer 源码树位置\n"
    f"  请设置环境变量 {NINFER_ROOT_ENV}=<ninfer 检出根目录>\n"
    "  该目录下应同时存在 tools/artifact 与 src/ops/linear/ternary。"
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


def ninfer_root() -> Path:
    """解析 ninfer 源码树根目录。

    Returns:
        ninfer 源码树根目录的绝对路径。

    Raises:
        SystemExit: 环境变量未设置，或所指向的目录不含 tools/artifact。
    """
    raw = os.environ.get(NINFER_ROOT_ENV)
    if not raw:
        raise SystemExit(_MISSING_HINT)
    root = Path(raw).expanduser().resolve()
    if not (root / "tools" / "artifact").is_dir():
        raise SystemExit(
            f"{NINFER_ROOT_ENV}={root} 下找不到 tools/artifact，"
            "请指向 ninfer 源码树根目录。"
        )
    return root


def bootstrap() -> Path:
    """挂好搜索路径并返回 ninfer 源码树根目录。

    ninfer 源码树最后被插入、因而排在最前，保证 tools 解析到它自带的常规包
    （本仓的 tools/ 是脚本目录，没有 __init__.py）。

    Returns:
        ninfer 源码树根目录的绝对路径。
    """
    root = ninfer_root()
    for entry in (repo_root(), tools_dir(), root):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root
