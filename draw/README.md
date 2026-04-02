# 折线图（draw/plot_latency.py）

从 benchmark 导出的 CSV 绘制折线图：横轴为 **batch_size** 或 **seq_len**，纵轴可选 **自动 / 耗时 / TFLOPS / 带宽**，同一张图内多条折线对应不同算子。

## 默认横轴 `auto`（推荐）

比较 CSV 里 **`batch_size` 与 `seq_len` 各自不同取值的个数**（`nunique`），**种类更多的那一维作为横轴**；相等时选 **batch**。仅存在一列时自动用该列。

运行时会打印一行：`[plot] x-axis auto -> ... (batch_size unique=..., seq_len unique=...)`。

`run.py --draw` 与 `plot_latency.py` 的 **`--x-axis` 默认均为 `auto`**。

## 默认纵轴 `auto`（推荐）

- **`decode`** 分组 → 纵轴 **带宽**（`bandwidth_gbs`）
- **`prefill_paged` / `prefill_ragged`** 分组 → 纵轴 **TFLOPS**（`tflops`）
- 其它 `op_type` → 纵轴 **耗时**（`latency_median_ms`）

`run.py --draw` 与 `plot_latency.py` 的 **`--y-metric` 默认均为 `auto`**。

## 依赖

与主项目一致，安装 `requirements.txt` 即可（含 `pandas`、`matplotlib`）：

```bash
pip install -r requirements.txt
```

## 用法

```bash
# 默认 auto：decode 用带宽、prefill 用 TFLOPS
python draw/plot_latency.py --csv results/results_20250101_120000.csv --x-axis batch

# 强制纵轴耗时 / TFLOPS / 带宽
python draw/plot_latency.py --csv results/results_20250101_120000.csv --y-metric latency
python draw/plot_latency.py --csv results/results_20250101_120000.csv --y-metric tflops
python draw/plot_latency.py --csv results/results_20250101_120000.csv --y-metric bandwidth_gbs

# 横轴 seq_len，只画 decode
python draw/plot_latency.py --csv results/results_20250101_120000.csv --x-axis seq_len --op-type decode --output-dir draw/out --format svg
```

与 `run.py` 联动：

```bash
python run.py --config configs/attention.yaml --draw
python run.py --config configs/attention.yaml --draw --draw-x-axis seq_len
python run.py --config configs/attention.yaml --draw --draw-y-metric latency
python run.py --config configs/attention.yaml --draw --draw-y-metric tflops --draw-x-axis seq_len
```

## 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--csv` | 输入的 CSV 文件路径（run.py 导出） | 必填 |
| `--x-axis` | `auto`（按唯一值多少选 batch/seq_len）或 `batch` / `seq_len` | `auto` |
| `--y-metric` | `auto` 或 `latency` / `tflops` / `bandwidth_gbs` | `auto` |
| `--output-dir` | 图片输出目录 | `draw/output` |
| `--op-type` | 只画指定类型：decode / prefill_paged / prefill_ragged | 全部 |
| `--format` | 图片格式：png / svg | png |

与 `run.py --draw` 联动：`--draw-x-axis`、`--draw-y-metric`（含 `auto`）。

## 输出

- 按「固定维度」分组，每组生成一张图。
- 文件名含指标 slug：`bandwidth_vs_batch_...`、`tflops_vs_batch_...`、`latency_vs_batch_...` 等。
- 各算子折线颜色/线型/标记按 **CSV 中 `operator` 列首次出现顺序** 固定映射，同一份 CSV 多次作图一致。

## 说明

- `tflops` / `bandwidth_gbs` 仅在 benchmark **该组精度全过** 时才会写入 CSV；`auto` 下某组若对应列为空则跳过该图。
- 若所有分组都被跳过，脚本会报错提示检查 CSV。
