#!/usr/bin/env python3
"""Decode-phase GPU time budget from an nsys report.

The decode step on this model is ~400 linear launches plus attention/GDN/norm work, so one aggregate
number hides which tensor dominates. This splits the report two ways: by kernel, and by GEMV output
width (GrdX * kGemvWarpsPerBlock, which identifies the projection), plus the per-token share.

Usage:
    nsys profile --trace=cuda --cuda-graph-trace=node -o trace <engine> <artifact> ...
    nsys stats --report cuda_gpu_sum -o stats trace.nsys-rep
    nsys stats --report cuda_gpu_trace --format csv -o trace_gpu trace.nsys-rep
    nsys_decode_budget.py stats_cuda_gpu_sum.csv [trace_gpu_cuda_gpu_trace.csv] [--tokens 32]

The kernel split needs only the first CSV, the per-shape split needs the second.

Caveat when reading the output: nsys inflates per-kernel durations somewhat (roughly 10-15% for the
100-400 us kernels here), so treat the per-shape numbers as upper bounds and compare against
gemv_bandwidth for the real rate.
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib

DURATION = "Duration (ns)"
TOTAL = "Total Time (ns)"
OPERATION = "Operation"
INSTANCES = "Instances"
NAME = "Name"
GRID_X = "GrdX"

# The decode GEMV assigns one warp per output row with 8 warps per block, so the block count is
# rows / 8. Keep in sync with kGemvWarpsPerBlock in ternary_rowsplit_gemv.cuh.
WARPS_PER_BLOCK = 8


def _read(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def kernel_split(path: pathlib.Path, top: int) -> None:
    rows = _read(path)
    rows.sort(key=lambda r: -float(r[TOTAL]))
    total = sum(float(r[TOTAL]) for r in rows)
    print(f"{'operation':<58} {'total ms':>9} {'launches':>9}")
    print("-" * 78)
    for row in rows[:top]:
        print(f"{row[OPERATION][:58]:<58} {float(row[TOTAL]) / 1e6:9.1f} {row[INSTANCES]:>9}")
    print("-" * 78)
    print(f"total GPU time: {total / 1e6:.1f} ms over {len(rows)} rows")

    def agg(predicate, label: str) -> None:
        selected = [r for r in rows if predicate(r[OPERATION])]
        if not selected:
            return
        subtotal = sum(float(r[TOTAL]) for r in selected)
        launches = sum(int(r[INSTANCES]) for r in selected)
        print(f"  {label:<34} {subtotal / 1e6:8.1f} ms {100.0 * subtotal / total:5.1f}%  "
              f"launches={launches}")

    print()
    agg(lambda n: "ternary_ptq1_gemv" in n, "PTQ1 decode GEMV")
    agg(lambda n: "ternary_ptq1_quantize_act" in n, "activation quantization")
    agg(lambda n: "rowsplit_gemm_kernel" in n, "T=8 GEMM (prefill)")
    agg(lambda n: "rotate" in n, "rotation")
    agg(lambda n: "gated_delta_net" in n, "GDN recurrent")
    agg(lambda n: "rmsnorm" in n, "rmsnorm")
    agg(lambda n: "MEMORY_OPER" in n or "memcpy" in n, "memcpy (model load)")


def shape_split(path: pathlib.Path, tokens: int) -> None:
    rows = _read(path)
    gemv: dict[int, list[int]] = collections.defaultdict(list)
    quant: list[int] = []
    for row in rows:
        duration = int(row.get(DURATION) or 0)
        name = row.get(NAME, "")
        if "ternary_ptq1_gemv" in name:
            gemv[int(row.get(GRID_X) or 0)].append(duration)
        elif "quantize_act" in name:
            quant.append(duration)
    if not gemv:
        print("trace CSV 里没有 PTQ1 GEMV 的行")
        return
    print(f"{'rows':>8} {'launches':>9} {'total ms':>10} {'median us':>10} {'ms/token':>10}")
    total = 0
    for grid, durations in sorted(gemv.items(), key=lambda kv: -len(kv[1])):
        durations.sort()
        subtotal = sum(durations)
        total += subtotal
        print(f"{grid * WARPS_PER_BLOCK:8d} {len(durations):9d} {subtotal / 1e6:10.1f} "
              f"{durations[len(durations) // 2] / 1e3:10.2f} {subtotal / 1e6 / tokens:10.2f}")
    print(f"GEMV 合计: {total / 1e6:.1f} ms -> {total / 1e6 / tokens:.2f} ms/token")
    if quant:
        quant.sort()
        print(f"quantize: {len(quant)} 次启动, {sum(quant) / 1e6:.1f} ms, "
              f"median {quant[len(quant) // 2] / 1e3:.2f} us -> "
              f"{sum(quant) / 1e6 / tokens:.2f} ms/token")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernel_csv", type=pathlib.Path, help="nsys cuda_gpu_sum CSV")
    parser.add_argument("trace_csv", nargs="?", type=pathlib.Path,
                        help="nsys cuda_gpu_trace CSV（per-shape 拆分需要）")
    parser.add_argument("--tokens", type=int, default=32,
                        help="被 profile 的解码步数（默认 32）")
    parser.add_argument("--top", type=int, default=16,
                        help="kernel 拆分的显示行数（默认 16）")
    args = parser.parse_args()

    kernel_split(args.kernel_csv, args.top)
    if args.trace_csv is not None:
        print()
        shape_split(args.trace_csv, args.tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
