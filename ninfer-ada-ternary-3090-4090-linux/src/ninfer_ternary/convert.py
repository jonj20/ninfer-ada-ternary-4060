"""ninfer-convert：把 Ternary Bonsai 的 GGUF 转换成三元 ninfer 制品。

真正的打包器是 tools/pack.py（自 ninfer-ada-ternary 平移，逐行可对照上游）。本模块只负责
把运行环境摆好：挂上打包器目录，并让"制品读写模块"在哪都能解析到 —— 仓库形态用
NINFER_ROOT 指向的 ninfer 检出，安装形态用随包分发的那份副本。
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

from . import assets


def _prepare() -> None:
    """摆好打包器需要的导入路径与上游根目录。

    Raises:
        SystemExit: 找不到打包器脚本，或缺少 numpy。
    """
    pack = assets.pack_dir()
    if not (pack / "pack.py").is_file():
        raise SystemExit(f"找不到打包器脚本: {pack / 'pack.py'}")
    entry = str(pack)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    upstream = assets.upstream_root()
    if upstream is not None:
        os.environ.setdefault(assets.UPSTREAM_ENV, str(upstream))


def main(argv: Sequence[str] | None = None) -> int:
    """命令行主入口。

    Args:
        argv: 参数列表；默认取 sys.argv[1:]。参数原样转交打包器。

    Returns:
        进程退出码。

    Raises:
        SystemExit: 缺少 numpy，或打包器自身以 SystemExit 收场。
    """
    args = list(sys.argv[1:] if argv is None else argv)
    _prepare()
    try:
        from pack import main as pack_main
    except ImportError as error:
        missing = getattr(error, "name", "") or str(error)
        raise SystemExit(
            f"转换器缺少依赖: {missing}\n"
            f"  打包器要读写张量，依赖 numpy 与 torch。只装 numpy 的那次安装不能转换模型，\n"
            f"  请按 README 的模型转换一节带上 convert 额外项安装：\n"
            f'    uv tool install --force "ninfer-ternary[convert]"'
        ) from error
    sys.argv = ["ninfer-convert", *args]
    return pack_main()
