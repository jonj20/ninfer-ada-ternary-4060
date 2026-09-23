"""Closed registry of persistent NInfer tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Signed grouped codes with one binary16 multiplier per group."""

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int


@dataclass(frozen=True, slots=True)
class TernaryFormat:
    """三值权重：base-3 三值码或 2-bit 码，每组一个 binary16 缩放。

    与 QuantFormat 的区别在于"每组多少字节"无法由位宽推出：Prism 三元用 128 宽的组，
    PTQ1_0 的基础平面是 24 B 且额外带 2 B 高位平面（5 个三值/字节，装不满整字节），
    PQ2_0 的基础平面是 32 B 且没有高位平面（4 个 2-bit 码/字节）。所以这里直接携带
    每组的平面字节数，而不是从 bits 反推。
    """

    name: str
    group_size: int
    base_bytes_per_group: int
    high_bytes_per_group: int


NumericFormat: TypeAlias = DirectFormat | QuantFormat | TernaryFormat


BF16 = DirectFormat("BF16", 2)
FP32 = DirectFormat("FP32", 4)
I32 = DirectFormat("I32", 4)

Q4G64_F16S = QuantFormat("Q4G64_F16S", 4, 64, -8, 7)
Q5G64_F16S = QuantFormat("Q5G64_F16S", 5, 64, -16, 15)
Q6G64_F16S = QuantFormat("Q6G64_F16S", 6, 64, -32, 31)
W8G32_F16S = QuantFormat("W8G32_F16S", 8, 32, -127, 127)

# Prism 私有三元（Ternary Bonsai 2 27B）。几何与引擎侧 artifact/storage_layouts.cpp 的
# quant_geometry() 必须逐字节一致：PTQ1_0 = 24 + 2，PQ2_0 = 32 + 0（每组 128 个权重）。
PTQ1_0_G128 = TernaryFormat("PTQ1_0_G128", 128, 24, 2)
PQ2_0_G128 = TernaryFormat("PQ2_0_G128", 128, 32, 0)


DIRECT_FORMATS = MappingProxyType(
    {item.name: item for item in (BF16, FP32, I32)}
)
QUANT_FORMATS = MappingProxyType(
    {
        item.name: item
        for item in (Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, W8G32_F16S)
    }
)
TERNARY_FORMATS = MappingProxyType(
    {item.name: item for item in (PTQ1_0_G128, PQ2_0_G128)}
)
NUMERIC_FORMATS = MappingProxyType(
    {**DIRECT_FORMATS, **QUANT_FORMATS, **TERNARY_FORMATS}
)


def valid_positive_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word represents a finite positive value."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value) and value > 0.0


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
    "DIRECT_FORMATS",
    "DirectFormat",
    "FP32",
    "I32",
    "NUMERIC_FORMATS",
    "NumericFormat",
    "PQ2_0_G128",
    "PTQ1_0_G128",
    "Q4G64_F16S",
    "Q5G64_F16S",
    "Q6G64_F16S",
    "QUANT_FORMATS",
    "QuantFormat",
    "TERNARY_FORMATS",
    "TernaryFormat",
    "W8G32_F16S",
    "get_format",
    "valid_positive_fp32_word",
]
