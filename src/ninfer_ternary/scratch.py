"""项目专属的临时目录根。

构建、导出、验证脚本产生的临时文件全部落在这个根下：清理只需删一个目录，
系统临时目录里也不会剩下认不出归属的碎片。根目录可用 NINFER_TERNARY_TMPDIR 覆盖。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

#: 覆盖临时根目录；默认是 $TMPDIR/ninfer-ternary。
TMP_ROOT_ENV = "NINFER_TERNARY_TMPDIR"

#: 临时根目录的目录名。
TMP_ROOT_NAME = "ninfer-ternary"


def prune_temp_root() -> None:
    """删掉空的默认根目录；显式覆盖过的目录一律不动。

    构建结束后根下通常空无一物，留个空壳目录会让人以为还有残留。
    """
    if os.environ.get(TMP_ROOT_ENV, "").strip():
        return
    root = Path(tempfile.gettempdir()) / TMP_ROOT_NAME
    try:
        root.rmdir()
    except OSError:
        # 还有别的临时树在跑，或目录已经被删掉，都不算异常。
        return


def temp_root() -> Path:
    """返回项目专属的临时根目录，不存在时创建。

    Returns:
        临时根目录；存在 NINFER_TERNARY_TMPDIR 时以它为准。

    Raises:
        OSError: 目录无法创建。
    """
    override = os.environ.get(TMP_ROOT_ENV, "").strip()
    root = Path(override).expanduser() if override else Path(tempfile.gettempdir()) / TMP_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root
