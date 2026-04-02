import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from tabulate import tabulate

from op_bench.term_style import green, red


class Reporter:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        op_type: str,
        params: Dict[str, Any],
        perf_results: Dict[str, Dict[str, Any]],
        accuracy_results: Dict[str, Dict[str, Any]],
    ):
        for op_name, acc in accuracy_results.items():
            row = {"op_type": op_type, **params, "operator": op_name}
            if op_name in perf_results:
                perf = perf_results[op_name]
                row["latency_median_ms"] = perf["latency_median_ms"]
                row["tflops"] = perf["tflops"]
                row["bandwidth_gbs"] = perf["bandwidth_gbs"]
            row["max_abs_diff"] = acc["max_abs_diff"]
            row["pass"] = acc["pass"]
            self._records.append(row)

    def print_table(self):
        if not self._records:
            print("No results recorded.")
            return

        print("\n" + "=" * 120)
        print("  BENCHMARK RESULTS (accuracy vs naive-torch baseline)")
        print("=" * 120)

        has_perf = any("latency_median_ms" in r for r in self._records)

        headers = [
            "op_type", "batch", "seq_len", "heads",
            "kv_heads", "dim", "operator",
        ]
        if has_perf:
            headers += ["median(ms)", "TFLOPS", "BW(GB/s)"]
        headers += ["max_diff", "status"]

        table = []
        for r in self._records:
            status = green("PASS") if r["pass"] else red("FAIL")
            row = [
                r.get("op_type", ""),
                r.get("batch_size", ""),
                r.get("seq_len", ""),
                r.get("num_heads", ""),
                r.get("num_kv_heads", ""),
                r.get("head_dim", ""),
                r.get("operator", ""),
            ]
            if has_perf:
                if "latency_median_ms" in r:
                    row += [
                        f"{r['latency_median_ms']:.4f}",
                        f"{r['tflops']:.2f}",
                        f"{r['bandwidth_gbs']:.1f}",
                    ]
                else:
                    row += ["-", "-", "-"]
            row += [f"{r['max_abs_diff']:.6f}", status]
            table.append(row)

        print(tabulate(table, headers=headers, tablefmt="grid"))

    def export(
        self,
        output_dir: str,
        export_csv: bool = True,
        export_json: bool = True,
    ) -> tuple:
        """Export records to CSV/JSON. Returns (csv_path or None, json_path or None)."""
        csv_path = None
        json_path = None
        if not export_csv and not export_json:
            return csv_path, json_path
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_csv:
            csv_path = os.path.join(output_dir, f"results_{ts}.csv")
            _write_csv(csv_path, self._records)
            print(f"\nCSV saved to: {csv_path}")

        if export_json:
            json_path = os.path.join(output_dir, f"results_{ts}.json")
            with open(json_path, "w") as f:
                json.dump(self._records, f, indent=2, default=str)
            print(f"JSON saved to: {json_path}")
        return csv_path, json_path


def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
