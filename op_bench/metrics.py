"""Compute theoretical FLOPs and memory bytes for attention operations.

FLOPs counting:
  - QK^T matmul: 2 * batch * num_heads * seq_q * seq_kv * head_dim
  - P @ V matmul: 2 * batch * num_heads * seq_q * seq_kv * head_dim
  - Total = 4 * batch * num_heads * seq_q * seq_kv * head_dim
  - For causal prefill, effective seq_kv per position averages seq_len/2,
    so total ~= 2 * batch * num_heads * seq_len^2 * head_dim

Memory bytes (bf16 = 2 bytes per element):
  - Counts all tensor reads/writes at the operator boundary.
"""

from op_bench.registry import OpType

BYTES_PER_ELEM = 2  # bf16


def compute_flops(op_type: OpType, batch_size: int, seq_len: int,
                  num_heads: int, num_kv_heads: int, head_dim: int, **_) -> int:
    if op_type == OpType.DECODE:
        # seq_q=1, seq_kv=seq_len, no causal reduction
        return 4 * batch_size * num_heads * seq_len * head_dim
    else:
        # causal prefill: average seq_kv ≈ seq_len / 2
        return 2 * batch_size * num_heads * seq_len * seq_len * head_dim


def compute_mem_bytes(op_type: OpType, batch_size: int, seq_len: int,
                      num_heads: int, num_kv_heads: int, head_dim: int, **_) -> int:
    if op_type == OpType.DECODE:
        # Q read + KV cache read + output write
        q_bytes = batch_size * num_heads * head_dim
        kv_bytes = 2 * batch_size * seq_len * num_kv_heads * head_dim
        out_bytes = batch_size * num_heads * head_dim
        return (q_bytes + kv_bytes + out_bytes) * BYTES_PER_ELEM
    else:
        # Q read + K read + V read + output write
        q_bytes = batch_size * seq_len * num_heads * head_dim
        kv_bytes = 2 * batch_size * seq_len * num_kv_heads * head_dim
        out_bytes = batch_size * seq_len * num_heads * head_dim
        return (q_bytes + kv_bytes + out_bytes) * BYTES_PER_ELEM


def compute_throughput(flops: int, mem_bytes: int, latency_ms: float):
    """Returns (tflops, bandwidth_gb_s) from raw counts and latency."""
    latency_s = latency_ms * 1e-3
    tflops = flops / latency_s / 1e12
    bw_gbs = mem_bytes / latency_s / 1e9
    return tflops, bw_gbs
