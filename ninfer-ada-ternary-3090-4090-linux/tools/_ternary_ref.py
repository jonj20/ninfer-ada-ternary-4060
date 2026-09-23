"""三元词表/权重的参考实现，供验证脚本复用。

三元格式（PQ2_0_G128 / PTQ1_0_G128）的 GGUF 读取与码解码只有一份实现，就在
`pack.py` 里。本模块只是把它重新导出，让 `tools/verify/` 下的脚本不必各自复制一份
解码器 —— 两份实现一旦漂移，"验证"就失去意义。

Example:
    from _ternary_ref import Gguf, dq_pq2_0, dq_ptq1_0
    g = Gguf("/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf")
"""

from __future__ import annotations

from pack import (  # noqa: F401
    FMT,
    T_BF16,
    T_F32,
    T_PQ2_0,
    T_PTQ1_0,
    Gguf,
    dq_pq2_0,
    dq_ptq1_0,
    read_direct,
    tiled_to_grouped,
)

__all__ = [
    "FMT",
    "Gguf",
    "T_BF16",
    "T_F32",
    "T_PQ2_0",
    "T_PTQ1_0",
    "dq_pq2_0",
    "dq_ptq1_0",
    "read_direct",
    "tiled_to_grouped",
]
