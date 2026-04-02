#!/usr/bin/env python3
"""Plot benchmark CSV: X-axis batch_size or seq_len; Y-axis latency / TFLOPS / bandwidth.

Reads the CSV produced by op_benchmark, groups by fixed dimensions,
and draws one line per operator per group.

Usage:
    python draw/plot_latency.py --csv results/results_20250101_120000.csv --x-axis batch
    python draw/plot_latency.py --csv results/results_20250101_120000.csv --y-metric tflops
    python draw/plot_latency.py --csv results/results_20250101_120000.csv --y-metric bandwidth_gbs
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# 图表样式：配色、线型、字体
LINE_COLORS = ["#2E86AB", "#E94F37", "#44AF69", "#FCAB10", "#7B2D8E", "#1B998B"]
LINE_STYLES = ["-"]
MARKERS = ["o", "s", "^", "D", "v", "p"]


def _setup_plot_style(ax):
    """统一设置坐标轴样式，使图表更清晰美观。"""
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.tick_params(axis="both", labelsize=10)
    ax.set_axisbelow(True)


# y_metric key -> (csv_column, y_axis_label, filename_slug)
Y_METRIC_CONFIG = {
    "latency": ("latency_median_ms", "Latency (ms)", "latency"),
    "tflops": ("tflops", "TFLOPS", "tflops"),
    "bandwidth_gbs": ("bandwidth_gbs", "Bandwidth (GB/s)", "bandwidth"),
}

# --y-metric auto: 按 op_type 选默认纵轴
AUTO_Y_BY_OP_TYPE = {
    "decode": "bandwidth_gbs",
    "prefill_paged": "tflops",
    "prefill_ragged": "tflops",
}

Y_METRIC_CHOICES = ("auto",) + tuple(Y_METRIC_CONFIG.keys())

X_AXIS_CHOICES = ("auto", "batch", "seq_len")


def _resolve_x_axis_auto(df: pd.DataFrame) -> str:
    """Pick x-axis by which of batch_size / seq_len has more distinct values; tie -> batch."""
    has_b = "batch_size" in df.columns
    has_s = "seq_len" in df.columns
    if not has_b and not has_s:
        raise ValueError("CSV needs batch_size and/or seq_len for plotting")
    if has_b and not has_s:
        return "batch"
    if has_s and not has_b:
        return "seq_len"
    nb = int(df["batch_size"].nunique(dropna=True))
    ns = int(df["seq_len"].nunique(dropna=True))
    if ns > nb:
        return "seq_len"
    return "batch"


def _operator_order_first_seen(df: pd.DataFrame) -> List[str]:
    """Operator names in order of first appearance in CSV rows (stable across runs)."""
    order: List[str] = []
    seen = set()
    for op in df["operator"]:
        if op not in seen:
            seen.add(op)
            order.append(op)
    return order


def _operator_style_index(op_order: List[str]) -> Dict[str, int]:
    """Map operator name -> index for color / line style / marker (fixed per name)."""
    return {name: i for i, name in enumerate(op_order)}


def _get_group_cols(x_axis: str, df: pd.DataFrame) -> list:
    """Columns to group by (fixed dimensions). Excludes x-axis and operator."""
    base = ["num_heads", "num_kv_heads", "head_dim"]
    if x_axis == "batch":
        base = ["op_type", "seq_len"] + base
    else:
        base = ["op_type", "batch_size"] + base
    if "page_size" in df.columns:
        base.append("page_size")
    return [c for c in base if c in df.columns]


def _plot_one_group(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_label: str,
    y_slug: str,
    x_axis: str,
    group_cols: list,
    group_key,
    op_order: List[str],
    op_to_idx: Dict[str, int],
    output_dir: str,
    format: str,
) -> bool:
    """Sort, draw, save. Returns True if a file was written."""
    sub = sub.sort_values(x_col)
    present = set(sub["operator"].unique())
    operators = [o for o in op_order if o in present]
    if not operators:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=100)
    _setup_plot_style(ax)

    for op in operators:
        rows = sub[sub["operator"] == op]
        idx = op_to_idx[op]
        color = LINE_COLORS[idx % len(LINE_COLORS)]
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot(
            rows[x_col].values,
            rows[y_col].values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
            markersize=7,
            markeredgewidth=1.5,
            markeredgecolor="white",
            label=op,
        )

    title = ", ".join(
        f"{c}={v}" for c, v in zip(group_cols, (group_key if isinstance(group_key, tuple) else (group_key,)))
    )
    ax.set_title(title, fontsize=12, fontweight="medium", pad=14)
    ax.set_xlabel("Batch size" if x_axis == "batch" else "Seq len", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=10, fancybox=True, framealpha=0.95)

    safe_name = "_".join(str(v) for v in (group_key if isinstance(group_key, tuple) else (group_key,)))
    fname = f"{y_slug}_vs_{x_axis}_{safe_name}.{format}"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return True


def run_plot(
    csv_path: str,
    output_dir: str = "draw/output",
    x_axis: str = "batch",
    y_metric: str = "auto",
    op_type: Optional[str] = None,
    format: str = "png",
) -> None:
    """Plot selected Y metric vs batch/seq_len from CSV. Can be called from run.py --draw.

    y_metric ``auto`` (default): decode -> bandwidth_gbs; prefill_paged / prefill_ragged -> tflops;
    unknown op_type -> latency.

    x_axis ``auto`` (default): compare ``batch_size`` vs ``seq_len`` unique counts; more categories wins; tie -> batch.
    """
    if y_metric not in Y_METRIC_CHOICES:
        raise ValueError(f"y_metric must be one of {Y_METRIC_CHOICES}")
    if x_axis not in X_AXIS_CHOICES:
        raise ValueError(f"x_axis must be one of {X_AXIS_CHOICES}")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "operator" not in df.columns or "op_type" not in df.columns:
        raise ValueError("CSV must contain 'operator' and 'op_type' columns")

    if op_type is not None:
        df = df[df["op_type"] == op_type]
        if df.empty:
            raise ValueError(f"No rows for op_type={op_type}")

    if x_axis == "auto":
        x_resolved = _resolve_x_axis_auto(df)
        nb = int(df["batch_size"].nunique(dropna=True)) if "batch_size" in df.columns else 0
        ns = int(df["seq_len"].nunique(dropna=True)) if "seq_len" in df.columns else 0
        print(
            f"[plot] x-axis auto -> {x_resolved} "
            f"(batch_size unique={nb}, seq_len unique={ns})"
        )
    else:
        x_resolved = x_axis

    x_col = "batch_size" if x_resolved == "batch" else "seq_len"
    if x_col not in df.columns:
        raise ValueError(f"CSV has no column '{x_col}' for x-axis={x_resolved}")
    group_cols = _get_group_cols(x_resolved, df)

    op_order = _operator_order_first_seen(df)
    op_to_idx = _operator_style_index(op_order)

    os.makedirs(output_dir, exist_ok=True)
    n_saved = 0

    if y_metric == "auto":
        for group_key, sub_raw in df.groupby(group_cols, dropna=False):
            ot = str(sub_raw["op_type"].iloc[0])
            eff_key = AUTO_Y_BY_OP_TYPE.get(ot, "latency")
            y_col, y_label, y_slug = Y_METRIC_CONFIG[eff_key]
            if y_col not in df.columns:
                continue
            sub = sub_raw.dropna(subset=[y_col])
            if sub.empty:
                continue
            if _plot_one_group(
                sub,
                x_col,
                y_col,
                y_label,
                y_slug,
                x_resolved,
                group_cols,
                group_key,
                op_order,
                op_to_idx,
                output_dir,
                format,
            ):
                n_saved += 1
    else:
        y_col, y_label, y_slug = Y_METRIC_CONFIG[y_metric]
        if y_col not in df.columns:
            raise ValueError(f"CSV has no column '{y_col}' (needed for --y-metric {y_metric})")

        df = df.dropna(subset=[y_col])
        if df.empty:
            raise ValueError(f"No rows with non-null '{y_col}'")

        for group_key, sub in df.groupby(group_cols, dropna=False):
            if _plot_one_group(
                sub,
                x_col,
                y_col,
                y_label,
                y_slug,
                x_resolved,
                group_cols,
                group_key,
                op_order,
                op_to_idx,
                output_dir,
                format,
            ):
                n_saved += 1

    if n_saved == 0:
        raise ValueError("No figures written (check CSV metrics / NaN rows for chosen y-metric)")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot latency / TFLOPS / bandwidth vs batch or seq_len from benchmark CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file (e.g. results/results_YYYYMMDD_HHMMSS.csv)",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        choices=list(X_AXIS_CHOICES),
        default="auto",
        help="X-axis: auto (more unique batch_size vs seq_len), or batch / seq_len",
    )
    parser.add_argument(
        "--y-metric",
        type=str,
        choices=list(Y_METRIC_CHOICES),
        default="auto",
        help="Y-axis: auto (decode->bandwidth_gbs, prefill->tflops), or latency/tflops/bandwidth_gbs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="draw/output",
        help="Directory for output plots (default: draw/output)",
    )
    parser.add_argument(
        "--op-type",
        type=str,
        choices=["decode", "prefill_paged", "prefill_ragged"],
        default=None,
        help="Only plot this op_type (default: all)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "svg"],
        default="png",
        help="Output image format (default: png)",
    )
    args = parser.parse_args()

    try:
        run_plot(
            csv_path=args.csv,
            output_dir=args.output_dir,
            x_axis=args.x_axis,
            y_metric=args.y_metric,
            op_type=args.op_type,
            format=args.format,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
