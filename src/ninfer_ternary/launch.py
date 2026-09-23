"""ninfer 与 ninfer-serve 的入口壳。

安装形态下这两个命令是 Python 壳，真正的 CUDA 引擎是随包分发的二进制。壳只做一件事：
把它 exec 起来，让参数、信号、退出码原样透传。
"""

from __future__ import annotations

import os
import sys

from . import assets


def _exec_engine(name: str) -> int:
    """把当前进程替换成指定的引擎可执行文件。

    Args:
        name: 可执行文件名，取 assets.ENGINE_PROGRAMS 中的一项。

    Returns:
        找不到可执行文件时的退出码；找到时本函数不返回。

    Raises:
        SystemExit: 可执行文件存在但无法执行。
    """
    binary = assets.engine_binary(name)
    if binary is None:
        print(
            f"找不到引擎可执行文件 {name}，两种可能：\n"
            f"  1) 本安装是只装 Python 侧的构建（NINFER_TERNARY_SKIP_BUILD=1）；用 \n"
            f"     uv tool install --force <本包> 重装即可。\n"
            f"  2) 引擎在别处编译过，用 NINFER_ENGINE_BIN=<可执行文件所在目录> 指过去。",
            file=sys.stderr,
        )
        return 2
    os.execv(str(binary), [str(binary), *sys.argv[1:]])
    return 0


def main_ninfer() -> int:
    """运行引擎命令行。

    Returns:
        进程退出码（成功 exec 时不返回）。
    """
    return _exec_engine("ninfer")


def main_serve() -> int:
    """运行推理服务。

    Returns:
        进程退出码（成功 exec 时不返回）。
    """
    return _exec_engine("ninfer-serve")
