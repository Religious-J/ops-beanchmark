# ops_benchmark

**Attention 算子基准测试**：在统一输入与同一套参数矩阵下，对比多种实现的**数值精度**（相对朴素 PyTorch baseline）与 **GPU 延迟 / 吞吐**（TFLOPS、有效带宽），并可选导出 CSV/JSON 与折线图。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| **注册机制** | `@register_operator(name=..., op_type=...)` 注册实现；支持可选 `setup` / `teardown`（计时外准备与清理） |
| **参数矩阵** | YAML 中为 `decode` / `prefill_paged` / `prefill_ragged` 分别配置列表，自动做笛卡尔积 |
| **精度** | 以 naive float32 baseline 为参考，`atol` 内比较最大绝对误差；任一算子未通过则该组**不测性能** |
| **性能** | 预热 + 重复测量取中位延迟；推导 TFLOPS、带宽等指标 |
| **报告** | 终端汇总表；`results/` 下带时间戳的 CSV、JSON |
| **绘图** | `run.py --draw` 或单独运行 `draw/plot_latency.py`（详见 [draw/README.md](draw/README.md)） |

---

## 环境要求

- Python 3
- **CUDA** 与 **PyTorch（GPU 版）**（benchmark 使用 `torch.cuda.synchronize()`）
- 各算子对应可选依赖（如 FlashAttention、FlashInfer、`sgl-kernel` 等）按你本地环境自行安装；未安装或导入失败的实现可能不参与或报错

---

## 安装

```bash
cd /path/to/op_benchmark
pip install -r requirements.txt
```

依赖：`torch`、`pyyaml`、`tabulate`、`pandas`、`matplotlib`（绘图用）。

---

## 快速开始

```bash
# 默认配置 configs/attention.yaml，结果写入 results/
python run.py

# 指定配置与输出目录
python run.py --config configs/attention.yaml --output-dir results

# 跑完后自动生成折线图（输出到 draw/output）
python run.py --draw

# !!!! 推荐使用
python run.py --config configs/test.yaml -v --draw
```

---

## 命令行（`run.py`）

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置路径，默认 `configs/attention.yaml` |
| `--output-dir` | CSV/JSON 输出目录，默认 `results` |
| `-v` / `--verbose` | 打印每组参数的详细精度与性能日志 |
| `--no-csv` / `--no-json` | 不导出对应格式 |
| `--draw` | 在导出 CSV 的前提下调用 `draw/plot_latency` |
| `--draw-x-axis` | `auto` \| `batch` \| `seq_len`，与绘图脚本一致 |
| `--draw-y-metric` | `auto` \| `latency` \| `tflops` \| `bandwidth_gbs` |

---

## 仓库结构

```
op_benchmark/
├── run.py                      # CLI 入口
├── requirements.txt
├── configs/
│   ├── attention.yaml          # 示例：全量参数矩阵 + 算子列表
│   └── test.yaml               # 可按需添加更小配置
├── op_bench/
│   ├── registry.py             # 注册表与 OpType
│   ├── inputs.py               # Decode / Prefill 输入 dataclass 与生成器
│   ├── baseline.py             # naive PyTorch baseline（精度参照）
│   ├── runner.py               # 加载配置、跑精度与测时
│   ├── reporter.py             # 表格与 CSV/JSON
│   ├── metrics.py              # FLOPs、访存、吞吐估算
│   └── operators/              # 各库适配（包内自动 import 子模块完成注册）
│       ├── hpc_ops.py
│       ├── flash_attn_ops.py
│       ├── flash_attn_v4_ops.py   # 需 sgl-kernel 等带 FA4 接口的环境
│       └── flashinfer_ops.py
├── draw/
│   ├── plot_latency.py
│   └── README.md
└── eazy_test/
    └── attention_op_test.py    # 历史/参考脚本
```

`runner` 会执行 `import op_bench.operators`，因此**在 `op_bench/operators/` 下新增模块**即可被自动加载并注册（模块名需能被 `pkgutil` 发现）。

---

## 算子类型（`op_type`）

| 值 | 含义 |
|----|------|
| `decode` | Decode：`q_seq_len=1`，Paged KV Cache |
| `prefill_paged` | Prefill + Paged KV Cache |
| `prefill_ragged` | Prefill + Ragged（连续）Q/K/V |

各类型对应的 YAML 参数键见下节；输入张量形状见 `op_bench/inputs.py` 中 `DecodeInputs` / `PrefillPagedInputs` / `PrefillRaggedInputs`。

---

## YAML 配置

**全局键**（与 `decode` 等并列，顶层）：

- `atol`：精度通过阈值（默认约 `0.016`）
- `warmup`：测时预热次数
- `repeat`：测时重复次数（取中位数）

**每个 `op_type` 小节**（如 `decode:`）可包含：

- 参数列表：`batch_size`、`seq_len`、`num_heads`、`num_kv_heads`、`head_dim`；Paged 类还需 `page_size`
- `operators`: 字符串列表，与 `@register_operator(name=...)` 的 `name` 一致；仅运行列出的实现

示例片段：

```yaml
atol: 0.016
warmup: 5
repeat: 20

decode:
  batch_size: [1, 8]
  seq_len: [128, 1024]
  num_heads: [8]
  num_kv_heads: [1]
  head_dim: [128]
  page_size: [64]
  operators: ["hpc", "sgl-flash_attn", "flashinfer"]
```

---

## 扩展：注册新算子

1. 在 `op_bench/operators/` 新增模块（例如 `my_ops.py`），会被自动导入。
2. 按类型实现 forward，签名接收对应 `inputs` dataclass，返回 `torch.Tensor`。

**简单注册**：

```python
import torch
from op_bench.registry import register_operator
from op_bench.inputs import DecodeInputs

@register_operator(name="my_lib", op_type="decode")
def my_decode(inputs: DecodeInputs) -> torch.Tensor:
    return my_lib.attention(
        q=inputs.q,
        k_cache=inputs.key_cache,
        v_cache=inputs.value_cache,
        # ... 按该库 API 填参
    )
```

**需要计时外 setup**（例如分配 workspace、`begin_forward`）：

```python
from op_bench.registry import register_operator

def my_setup(inputs):
    inputs._wrapper = build_wrapper(inputs)

@register_operator(name="my_lib", op_type="decode", setup=my_setup, teardown=my_teardown)
def my_decode(inputs):
    return inputs._wrapper.forward(...)
```

3. 在 YAML 对应小节把 `"my_lib"` 加入 `operators` 列表。

---

## 输出说明

- **终端**：默认进度行 + 结束时的汇总表；`-v` 可查看每组 baseline/算子对比与延迟。
- **`results/`**：时间戳命名的 CSV、JSON，便于脚本或 `pandas` 再分析。
- **精度未全过**：该参数组合下**不记录性能**；CSV 中 TFLOPS/带宽等可能为空，绘图脚本会按 [draw/README.md](draw/README.md) 说明跳过或报错。

---

## 绘图

```bash
python run.py --config configs/attention.yaml --draw
python run.py --draw --draw-x-axis seq_len --draw-y-metric latency
```

或直接使用：

```bash
python draw/plot_latency.py --csv results/results_YYYYMMDD_HHMMSS.csv
```

默认横轴、纵轴的 `auto` 规则与可选参数见 **[draw/README.md](draw/README.md)**。

---

## Roadmap
- 支持 Gemm 类算子
