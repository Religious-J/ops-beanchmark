#!/usr/bin/env python3
"""H20 GPU hardware benchmark — memory bandwidth & compute throughput.

Tests:
  1. Memory bandwidth (STREAM-style): copy / scale / add / triad
  2. Compute throughput (GEMM): FP32, TF32, BF16, FP16, FP8-E4M3, INT8

Results are compared against NVIDIA H20 theoretical peak specs (dense, no sparsity):
  - HBM3 bandwidth: 4.0 TB/s
  - FP32:           44  TFLOPS
  - TF32:           74  TFLOPS  (Tensor Core, dense)
  - BF16/FP16:      148 TFLOPS  (Tensor Core, dense)
  - FP8:            296 TFLOPS  (Tensor Core, dense)
  - INT8:           296 TOPS    (Tensor Core, dense)

Usage:
    python hw_benchmark.py [--warmup 10] [--repeat 50] [--verbose]
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from tabulate import tabulate


# ---------------------------------------------------------------------------
# H20 theoretical peak specs
# ---------------------------------------------------------------------------
H20_SPECS = {
    "gpu_name":          "NVIDIA H20",
    "memory_gb":         96,
    "hbm_bw_tb_s":       4.0,       # TB/s
    "fp32_tflops":       44.0,
    "tf32_tflops":       74.0,      # dense (148 with sparsity)
    "bf16_tflops":       148.0,     # dense (296 with sparsity)
    "fp16_tflops":       148.0,     # dense (296 with sparsity)
    "fp8_tflops":        296.0,     # dense (592 with sparsity)
    "int8_tops":         296.0,     # dense (592 with sparsity)
}


@dataclass
class BenchResult:
    name: str
    dtype: str
    size_desc: str
    metric_value: float   # GB/s or TFLOPS
    metric_unit: str      # "GB/s" or "TFLOPS"
    peak_value: float     # theoretical peak in same unit
    utilization: float    # metric_value / peak_value * 100
    latency_us: float     # median latency in microseconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sync():
    torch.cuda.synchronize()


def _median_latency_us(fn, warmup: int, repeat: int) -> float:
    """Run *fn*, return median wall-clock time in microseconds."""
    for _ in range(warmup):
        fn()
    _sync()

    times: List[float] = []
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]


def _bytes_to_gb(b: int) -> float:
    return b / 1e9


# ---------------------------------------------------------------------------
# 1. Memory bandwidth benchmark (STREAM-style)
# ---------------------------------------------------------------------------
STREAM_SIZES = [
    ("256 MB",  256 * 1024 * 1024),
    ("1 GB",   1024 * 1024 * 1024),
    ("4 GB",   4 * 1024 * 1024 * 1024),
    ("16 GB", 16 * 1024 * 1024 * 1024),
]


def _stream_copy(a: torch.Tensor, b: torch.Tensor):
    """c[:] = a[:]  — 1 read + 1 write"""
    b.copy_(a)


def _stream_scale(a: torch.Tensor, b: torch.Tensor, scalar: float = 3.0):
    """b[:] = scalar * a[:]  — 1 read + 1 write"""
    torch.mul(a, scalar, out=b)


def _stream_add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    """c[:] = a[:] + b[:]  — 2 reads + 1 write"""
    torch.add(a, b, out=c)


def _stream_triad(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                  scalar: float = 3.0):
    """a[:] = b[:] + scalar * c[:]  — 2 reads + 1 write"""
    torch.addcmul(b, torch.ones_like(c), c, value=scalar, out=a)


def run_bandwidth_tests(warmup: int, repeat: int, verbose: bool) -> List[BenchResult]:
    peak_bw_gbs = H20_SPECS["hbm_bw_tb_s"] * 1000  # convert TB/s → GB/s
    results: List[BenchResult] = []

    tests = [
        ("Copy",  2, lambda a, b, c: _stream_copy(a, b)),
        ("Scale", 2, lambda a, b, c: _stream_scale(a, b)),
        ("Add",   3, lambda a, b, c: _stream_add(a, b, c)),
        ("Triad", 3, lambda a, b, c: _stream_triad(a, b, c)),
    ]

    for size_label, n_bytes in STREAM_SIZES:
        n_elems = n_bytes // 4  # float32 = 4 bytes
        a = torch.empty(n_elems, dtype=torch.float32, device="cuda")
        b = torch.empty_like(a)
        c = torch.empty_like(a)

        a.normal_()
        b.normal_()
        c.normal_()

        for test_name, rw_streams, fn in tests:
            total_bytes = rw_streams * n_bytes
            lat_us = _median_latency_us(lambda: fn(a, b, c), warmup, repeat)
            bw_gbs = _bytes_to_gb(total_bytes) / (lat_us * 1e-6)
            util = bw_gbs / peak_bw_gbs * 100

            results.append(BenchResult(
                name=test_name,
                dtype="float32",
                size_desc=size_label,
                metric_value=bw_gbs,
                metric_unit="GB/s",
                peak_value=peak_bw_gbs,
                utilization=util,
                latency_us=lat_us,
            ))

            if verbose:
                print(f"  BW  {test_name:6s}  {size_label:>6s}  "
                      f"{bw_gbs:8.1f} GB/s  ({util:5.1f}%)  lat={lat_us:.1f} us")

        del a, b, c
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 2. Compute throughput benchmark (GEMM)
# ---------------------------------------------------------------------------
GEMM_SIZES = [
    (1024,  1024,  1024),
    (2048,  2048,  2048),
    (4096,  4096,  4096),
    (8192,  8192,  8192),
    (16384, 16384, 16384),
]

GEMM_CONFIGS = [
    # (label, dtype, tensor_core_math, peak_tflops)
    ("FP32",          torch.float32,    None,                        H20_SPECS["fp32_tflops"]),
    ("TF32",          torch.float32,    torch.float32,               H20_SPECS["tf32_tflops"]),
    ("BF16",          torch.bfloat16,   None,                        H20_SPECS["bf16_tflops"]),
    ("FP16",          torch.float16,    None,                        H20_SPECS["fp16_tflops"]),
]


def _gemm_flops(M: int, N: int, K: int) -> int:
    return 2 * M * N * K


def run_gemm_tests(warmup: int, repeat: int, verbose: bool) -> List[BenchResult]:
    results: List[BenchResult] = []

    for label, dtype, tc_math, peak in GEMM_CONFIGS:
        for M, N, K in GEMM_SIZES:
            try:
                a = torch.randn(M, K, dtype=dtype, device="cuda")
                b = torch.randn(K, N, dtype=dtype, device="cuda")
            except Exception:
                continue

            flops = _gemm_flops(M, N, K)

            if tc_math is not None:
                ctx = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                if dtype == torch.float32:
                    ctx = torch.backends.cuda.matmul.allow_tf32
                    torch.backends.cuda.matmul.allow_tf32 = False

            lat_us = _median_latency_us(lambda: torch.mm(a, b), warmup, repeat)

            if dtype == torch.float32:
                torch.backends.cuda.matmul.allow_tf32 = ctx

            tflops = flops / (lat_us * 1e-6) / 1e12
            util = tflops / peak * 100

            size_desc = f"{M}x{N}x{K}"
            results.append(BenchResult(
                name=f"GEMM-{label}",
                dtype=label,
                size_desc=size_desc,
                metric_value=tflops,
                metric_unit="TFLOPS",
                peak_value=peak,
                utilization=util,
                latency_us=lat_us,
            ))

            if verbose:
                print(f"  GEMM {label:5s}  {size_desc:>17s}  "
                      f"{tflops:8.2f} TFLOPS  ({util:5.1f}%)  lat={lat_us:.1f} us")

            del a, b
            torch.cuda.empty_cache()

    return results


def run_fp8_gemm_tests(warmup: int, repeat: int, verbose: bool) -> List[BenchResult]:
    """FP8 GEMM using torch._scaled_mm (available on Hopper+)."""
    results: List[BenchResult] = []
    peak = H20_SPECS["fp8_tflops"]

    if not hasattr(torch, "float8_e4m3fn"):
        if verbose:
            print("  [SKIP] FP8 dtype not available in this PyTorch build")
        return results

    for M, N, K in GEMM_SIZES:
        try:
            a_fp32 = torch.randn(M, K, device="cuda")
            b_fp32 = torch.randn(N, K, device="cuda")  # transposed for _scaled_mm

            a = a_fp32.to(torch.float8_e4m3fn)
            b = b_fp32.to(torch.float8_e4m3fn)
            del a_fp32, b_fp32

            scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
            scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)

            def _fp8_mm():
                torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b,
                                 out_dtype=torch.bfloat16)

            _fp8_mm()
        except Exception as e:
            if verbose:
                print(f"  [SKIP] FP8 GEMM {M}x{N}x{K}: {e}")
            continue

        flops = _gemm_flops(M, N, K)
        lat_us = _median_latency_us(_fp8_mm, warmup, repeat)
        tflops = flops / (lat_us * 1e-6) / 1e12
        util = tflops / peak * 100
        size_desc = f"{M}x{N}x{K}"

        results.append(BenchResult(
            name="GEMM-FP8",
            dtype="FP8-E4M3",
            size_desc=size_desc,
            metric_value=tflops,
            metric_unit="TFLOPS",
            peak_value=peak,
            utilization=util,
            latency_us=lat_us,
        ))

        if verbose:
            print(f"  GEMM FP8    {size_desc:>17s}  "
                  f"{tflops:8.2f} TFLOPS  ({util:5.1f}%)  lat={lat_us:.1f} us")

        del a, b
        torch.cuda.empty_cache()

    return results


def run_int8_gemm_tests(warmup: int, repeat: int, verbose: bool) -> List[BenchResult]:
    """INT8 GEMM via torch._int_mm (Hopper+)."""
    results: List[BenchResult] = []
    peak = H20_SPECS["int8_tops"]

    for M, N, K in GEMM_SIZES:
        try:
            a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
            b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="cuda")

            def _int8_mm():
                torch._int_mm(a, b)

            _int8_mm()
        except Exception as e:
            if verbose:
                print(f"  [SKIP] INT8 GEMM {M}x{N}x{K}: {e}")
            continue

        ops = _gemm_flops(M, N, K)
        lat_us = _median_latency_us(_int8_mm, warmup, repeat)
        tops = ops / (lat_us * 1e-6) / 1e12
        util = tops / peak * 100
        size_desc = f"{M}x{N}x{K}"

        results.append(BenchResult(
            name="GEMM-INT8",
            dtype="INT8",
            size_desc=size_desc,
            metric_value=tops,
            metric_unit="TOPS",
            peak_value=peak,
            utilization=util,
            latency_us=lat_us,
        ))

        if verbose:
            print(f"  GEMM INT8   {size_desc:>17s}  "
                  f"{tops:8.2f} TOPS   ({util:5.1f}%)  lat={lat_us:.1f} us")

        del a, b
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_gpu_info():
    props = torch.cuda.get_device_properties(0)
    print(f"\n{'='*72}")
    print(f"  GPU Hardware Benchmark")
    print(f"{'='*72}")
    print(f"  Device:            {props.name}")
    print(f"  Compute Cap:       {props.major}.{props.minor}")
    print(f"  SMs:               {props.multi_processor_count}")
    print(f"  Memory:            {props.total_memory / 1024**3:.1f} GB")
    print(f"  PyTorch:           {torch.__version__}")
    print(f"  CUDA:              {torch.version.cuda}")
    print(f"{'='*72}")

    print(f"\n  H20 Theoretical Peaks (dense, no sparsity):")
    print(f"    HBM3 Bandwidth:  {H20_SPECS['hbm_bw_tb_s']} TB/s  ({H20_SPECS['hbm_bw_tb_s']*1000:.0f} GB/s)")
    print(f"    FP32:            {H20_SPECS['fp32_tflops']} TFLOPS")
    print(f"    TF32:            {H20_SPECS['tf32_tflops']} TFLOPS  (Tensor Core)")
    print(f"    BF16/FP16:       {H20_SPECS['bf16_tflops']} TFLOPS  (Tensor Core)")
    print(f"    FP8:             {H20_SPECS['fp8_tflops']} TFLOPS  (Tensor Core)")
    print(f"    INT8:            {H20_SPECS['int8_tops']} TOPS    (Tensor Core)")
    print()


def print_results(title: str, results: List[BenchResult]):
    if not results:
        return

    print(f"\n  {title}")
    print(f"  {'-'*len(title)}")

    headers = ["Test", "Dtype", "Size", "Throughput", "Unit", "Peak", "Util %", "Latency (us)"]
    rows = []
    for r in results:
        rows.append([
            r.name,
            r.dtype,
            r.size_desc,
            f"{r.metric_value:.1f}",
            r.metric_unit,
            f"{r.peak_value:.0f}",
            f"{r.utilization:.1f}%",
            f"{r.latency_us:.1f}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="simple", stralign="right"))


def print_summary(bw_results: List[BenchResult], compute_results: List[BenchResult]):
    """Print a concise summary showing the best achieved numbers."""
    print(f"\n{'='*72}")
    print(f"  Summary — Peak Achieved")
    print(f"{'='*72}")

    if bw_results:
        best_bw = max(bw_results, key=lambda r: r.metric_value)
        print(f"  Memory Bandwidth:  {best_bw.metric_value:.1f} GB/s  "
              f"({best_bw.utilization:.1f}% of {best_bw.peak_value:.0f} GB/s)  "
              f"[{best_bw.name}, {best_bw.size_desc}]")

    dtype_best = {}
    for r in compute_results:
        key = r.dtype
        if key not in dtype_best or r.metric_value > dtype_best[key].metric_value:
            dtype_best[key] = r

    for dtype_label in ["FP32", "TF32", "BF16", "FP16", "FP8-E4M3", "INT8"]:
        if dtype_label not in dtype_best:
            continue
        r = dtype_best[dtype_label]
        unit = r.metric_unit
        print(f"  {dtype_label:10s} Compute: {r.metric_value:.1f} {unit}  "
              f"({r.utilization:.1f}% of {r.peak_value:.0f} {unit})  "
              f"[{r.size_desc}]")

    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="H20 GPU hardware benchmark — bandwidth & compute throughput")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=50,
                        help="Timed iterations (default: 50)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-test progress")
    parser.add_argument("--skip-bw", action="store_true",
                        help="Skip bandwidth tests")
    parser.add_argument("--skip-compute", action="store_true",
                        help="Skip compute tests")
    args = parser.parse_args()

    print_gpu_info()

    bw_results: List[BenchResult] = []
    compute_results: List[BenchResult] = []

    if not args.skip_bw:
        print("  [1/3] Running memory bandwidth tests (STREAM-style) ...")
        bw_results = run_bandwidth_tests(args.warmup, args.repeat, args.verbose)
        print_results("Memory Bandwidth", bw_results)

    if not args.skip_compute:
        print("\n  [2/3] Running GEMM compute tests (FP32 / TF32 / BF16 / FP16) ...")
        gemm_results = run_gemm_tests(args.warmup, args.repeat, args.verbose)
        compute_results.extend(gemm_results)
        print_results("GEMM Compute Throughput", gemm_results)

        print("\n  [3/3] Running GEMM compute tests (FP8 / INT8) ...")
        fp8_results = run_fp8_gemm_tests(args.warmup, args.repeat, args.verbose)
        compute_results.extend(fp8_results)
        if fp8_results:
            print_results("FP8 Compute Throughput", fp8_results)

        int8_results = run_int8_gemm_tests(args.warmup, args.repeat, args.verbose)
        compute_results.extend(int8_results)
        if int8_results:
            print_results("INT8 Compute Throughput", int8_results)

    print_summary(bw_results, compute_results)


if __name__ == "__main__":
    main()
