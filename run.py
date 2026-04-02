#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from op_bench.runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Attention operator benchmark framework")
    parser.add_argument(
        "--config", type=str, default="configs/attention.yaml",
        help="Path to YAML config file (default: configs/attention.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for CSV/JSON output (default: results)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print detailed per-parameter logs (default: quiet, only summary table)",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Do not export results to CSV file",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Do not export results to JSON file",
    )
    parser.add_argument(
        "--draw", action="store_true",
        help="After benchmark, run draw/plot_latency and save plots to draw/output",
    )
    parser.add_argument(
        "--draw-y-metric",
        type=str,
        choices=["auto", "latency", "tflops", "bandwidth_gbs"],
        default="auto",
        help="With --draw: auto=decode->bandwidth_gbs, prefill->tflops; or force latency/tflops/bandwidth_gbs",
    )
    parser.add_argument(
        "--draw-x-axis",
        type=str,
        choices=["auto", "batch", "seq_len"],
        default="auto",
        help="With --draw: auto picks axis with more unique batch_size vs seq_len; or batch / seq_len",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config file '{args.config}' not found.")
        sys.exit(1)

    run_benchmark(
        args.config,
        args.output_dir,
        verbose=args.verbose,
        export_csv=not args.no_csv,
        export_json=not args.no_json,
        draw=args.draw,
        draw_y_metric=args.draw_y_metric,
        draw_x_axis=args.draw_x_axis,
    )


if __name__ == "__main__":
    main()
