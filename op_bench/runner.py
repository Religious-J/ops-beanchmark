import itertools
import time
import traceback
from typing import Any, Callable, Dict, List, Tuple

from op_bench.term_style import green, red

import torch
import yaml

from op_bench.registry import OpType, RegisteredOp, get_registry
from op_bench.inputs import (
    generate_decode_inputs,
    generate_prefill_paged_inputs,
    generate_prefill_ragged_inputs,
)
from op_bench.baseline import (
    baseline_decode,
    baseline_prefill_paged,
    baseline_prefill_ragged,
)
from op_bench.metrics import compute_flops, compute_mem_bytes, compute_throughput
from op_bench.reporter import Reporter

INPUT_GENERATORS = {
    OpType.DECODE: generate_decode_inputs,
    OpType.PREFILL_PAGED: generate_prefill_paged_inputs,
    OpType.PREFILL_RAGGED: generate_prefill_ragged_inputs,
}

BASELINE_FNS = {
    OpType.DECODE: baseline_decode,
    OpType.PREFILL_PAGED: baseline_prefill_paged,
    OpType.PREFILL_RAGGED: baseline_prefill_ragged,
}

_PARAM_KEYS = {
    OpType.DECODE: ["batch_size", "seq_len", "num_heads", "num_kv_heads", "head_dim", "page_size"],
    OpType.PREFILL_PAGED: ["batch_size", "seq_len", "num_heads", "num_kv_heads", "head_dim", "page_size"],
    OpType.PREFILL_RAGGED: ["batch_size", "seq_len", "num_heads", "num_kv_heads", "head_dim"],
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def iter_param_combinations(section: Dict[str, Any], param_keys: List[str]):
    lists = []
    for key in param_keys:
        val = section.get(key, [])
        if not isinstance(val, list):
            val = [val]
        lists.append(val)

    for combo in itertools.product(*lists):
        yield dict(zip(param_keys, combo))


def measure_latency(
    op: RegisteredOp,
    inputs: Any,
    warmup: int = 5,
    repeat: int = 20,
) -> Tuple[torch.Tensor, float]:
    """Run the operator with warmup, then measure latency.

    Returns (output, median_ms).
    """
    if op.setup is not None:
        op.setup(inputs)

    fn = op.forward

    for _ in range(warmup):
        output = fn(inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = fn(inputs)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

    if op.teardown is not None:
        op.teardown(inputs)

    times.sort()
    median = times[len(times) // 2]
    return output, median


def _print_tensor_stats(label: str, t: torch.Tensor) -> None:
    """One-line stats for verbose accuracy debugging."""
    f = t.detach().float().reshape(-1)
    print(
        f"    [compare] {label}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={f.min().item():.6f} max={f.max().item():.6f} mean={f.mean().item():.6f}"
    )


def _print_op_vs_baseline(baseline: torch.Tensor, op_out: torch.Tensor) -> None:
    """Where the largest deviation is (verbose)."""
    b = baseline.detach().float().reshape(-1)
    o = op_out.detach().float().reshape(-1)
    if b.numel() != o.numel():
        print(
            f"    [compare] shape mismatch baseline={tuple(baseline.shape)} "
            f"op={tuple(op_out.shape)}"
        )
        return
    diff = (o - b).abs()
    idx = int(diff.argmax().item())
    print(
        f"    [compare] worst flat_index={idx} baseline={b[idx].item():.6f} "
        f"op={o[idx].item():.6f} abs_diff={diff[idx].item():.6f}"
    )


def check_accuracy_vs_baseline(
    op: RegisteredOp,
    inputs: Any,
    baseline_output: torch.Tensor,
    atol: float,
) -> Dict[str, Any]:
    """Run operator once and compare against baseline output."""
    if op.setup is not None:
        op.setup(inputs)

    output = op.forward(inputs)
    torch.cuda.synchronize()

    out_flat = output.float().view(-1)
    ref_flat = baseline_output.float().view(-1)
    max_diff = (out_flat - ref_flat).abs().max().item()
    passed = max_diff <= atol

    return {"output": output, "max_abs_diff": max_diff, "pass": passed}


def _count_combinations(config: Dict[str, Any]) -> int:
    """Count total parameter combinations across all op types."""
    total = 0
    for op_type in OpType:
        section = config.get(op_type.value)
        if section is None:
            continue
        param_keys = _PARAM_KEYS[op_type]
        n = 1
        for key in param_keys:
            val = section.get(key, [])
            n *= len(val) if isinstance(val, list) else 1
        total += n
    return total


def run_benchmark(
    config_path: str,
    output_dir: str,
    verbose: bool = False,
    export_csv: bool = True,
    export_json: bool = True,
    draw: bool = False,
    draw_y_metric: str = "auto",
    draw_x_axis: str = "auto",
):
    """Main entry: load config, run all benchmarks, report results. If draw=True and CSV was exported, run plot_latency."""
    config = load_config(config_path)
    atol = config.get("atol", 0.016)
    warmup = config.get("warmup", 5)
    repeat = config.get("repeat", 20)

    import op_bench.operators  # noqa: F401

    registry = get_registry()
    reporter = Reporter()

    total_combos = _count_combinations(config)
    done = 0
    failed = 0
    passed = 0

    for op_type in OpType:
        section = config.get(op_type.value)
        if section is None:
            continue

        requested_ops = section.get("operators")
        available_ops = registry[op_type]

        if requested_ops is not None:
            ops_to_run = {
                name: op for name, op in available_ops.items() if name in requested_ops
            }
        else:
            ops_to_run = available_ops

        if not ops_to_run:
            if verbose:
                print(f"[SKIP] No operators registered for '{op_type.value}'")
            continue

        param_keys = _PARAM_KEYS[op_type]
        gen_fn = INPUT_GENERATORS[op_type]
        baseline_fn = BASELINE_FNS[op_type]

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Benchmarking: {op_type.value}")
            print(f"  Operators: {list(ops_to_run.keys())}")
            print(f"{'='*60}")

        for params in iter_param_combinations(section, param_keys):
            done += 1
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())

            if verbose:
                print(f"\n  [{op_type.value}] {param_str}")
            else:
                print(f"\r  [{done}/{total_combos}] {op_type.value}: {param_str}",
                      end="", flush=True)

            inputs = gen_fn(**params)

            flops = compute_flops(op_type, **params)
            mem_bytes = compute_mem_bytes(op_type, **params)

            # --- Phase 1: compute baseline (naive torch, float32) ---
            if verbose:
                print("    baseline         computing...")
            baseline_output = baseline_fn(inputs)
            torch.cuda.synchronize()

            # --- Phase 2: accuracy check for each operator vs baseline ---
            accuracy_results = {}
            all_passed = True
            for name, registered_op in ops_to_run.items():
                try:
                    acc = check_accuracy_vs_baseline(
                        registered_op, inputs, baseline_output, atol
                    )
                    accuracy_results[name] = acc
                    status = green("PASS") if acc["pass"] else red("FAIL")
                    if acc["pass"]:
                        passed += 1
                    if verbose:
                        print(f"    {name:15s}  max_diff={acc['max_abs_diff']:.6f} [{status}]")
                    if not acc["pass"]:
                        all_passed = False
                        failed += 1
                        if not verbose:
                            print(
                                f"\n    {red('FAIL')}: {name} max_diff={acc['max_abs_diff']:.6f}"
                            )
                        if verbose:
                            _print_tensor_stats("baseline", baseline_output)
                            _print_tensor_stats(f"op({name})", acc["output"])
                            _print_op_vs_baseline(baseline_output, acc["output"])
                except Exception as e:
                    all_passed = False
                    failed += 1
                    print(f"\n    {red('ACCURACY ERROR')} ({name}): {e}")
                    if verbose:
                        print("    [compare] traceback:")
                        for line in traceback.format_exc().splitlines():
                            print(f"      {line}")
                        _print_tensor_stats("baseline (reference)", baseline_output)

            # --- Phase 3: performance benchmark (only if accuracy passed) ---
            perf_results = {}
            if all_passed:
                for name, registered_op in ops_to_run.items():
                    try:
                        _, med = measure_latency(
                            registered_op, inputs, warmup, repeat
                        )
                        tflops, bw_gbs = compute_throughput(flops, mem_bytes, med)
                        perf_results[name] = {
                            "latency_median_ms": med,
                            "tflops": tflops,
                            "bandwidth_gbs": bw_gbs,
                        }
                        if verbose:
                            print(f"    {name:15s}  {med:.4f}ms  {tflops:.2f} TFLOPS  {bw_gbs:.1f} GB/s")
                    except Exception as e:
                        print(f"\n    {red('PERF ERROR')} ({name}): {e}")
            else:
                if verbose:
                    print("    [SKIP PERF] accuracy check failed")

            reporter.record(op_type.value, params, perf_results, accuracy_results)

            del inputs, baseline_output, accuracy_results
            torch.cuda.empty_cache()

    if not verbose:
        print()  # newline after progress

    reporter.print_table()
    print(f"\n  total: {green(f'PASS {passed}')}, {red(f'FAIL {failed}')}")
    csv_path, _ = reporter.export(output_dir, export_csv=export_csv, export_json=export_json)

    if draw:
        if csv_path is None:
            print("\n[--draw] Skipped: no CSV exported (run without --no-csv to enable draw).")
        else:
            try:
                from draw.plot_latency import run_plot
                y_hint = draw_y_metric
                if draw_y_metric == "auto":
                    y_hint = "auto (decode->BW, prefill->TFLOPS)"
                x_hint = draw_x_axis
                if draw_x_axis == "auto":
                    x_hint = "auto (more unique batch_size vs seq_len)"
                print(
                    f"\n[--draw] Generating plots (y={y_hint}, x-axis={x_hint}, output=draw/output)..."
                )
                run_plot(
                    csv_path=csv_path,
                    output_dir="draw/output",
                    x_axis=draw_x_axis,
                    y_metric=draw_y_metric,
                )
            except Exception as e:
                print(f"\n[--draw] Skipped: {e}", flush=True)
