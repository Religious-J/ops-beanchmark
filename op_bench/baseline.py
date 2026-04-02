"""Naive PyTorch attention baselines for accuracy reference.

All computations are done in float32 to serve as the ground-truth.
Adapted from torch_test/test_decode_bf16.py, test_with_kvcache_prefill_bf16.py,
and test_prefill_bf16.py.
"""

import math

import torch
import torch.nn.functional as F

from op_bench.inputs import DecodeInputs, PrefillPagedInputs, PrefillRaggedInputs


def baseline_decode(inputs: DecodeInputs) -> torch.Tensor:
    """Naive decode attention (q_seq_len=1) with paged KV cache.

    Gathers K/V from paged cache via block_ids, computes standard
    scaled-dot-product attention in float32, returns bf16 output.
    """
    batch_size = inputs.batch_size
    num_heads = inputs.num_heads
    num_kv_heads = inputs.num_kv_heads
    head_dim = inputs.head_dim
    page_size = inputs.page_size
    seq_len = inputs.seq_len
    groups = num_heads // num_kv_heads

    q = inputs.q.float()  # [batch_size, num_heads, head_dim]
    output = torch.empty_like(q)

    num_pages_per_seq = (seq_len + page_size - 1) // page_size

    for bi in range(batch_size):
        q_i = q[bi].unsqueeze(1)  # [num_heads, 1, head_dim]

        blk_ids = inputs.block_ids[bi, :num_pages_per_seq]
        k_pages = inputs.key_cache[blk_ids]    # [pages, page_size, num_kv_heads, head_dim]
        v_pages = inputs.value_cache[blk_ids]

        k_flat = k_pages.reshape(-1, num_kv_heads, head_dim)[:seq_len].float()
        v_flat = v_pages.reshape(-1, num_kv_heads, head_dim)[:seq_len].float()

        # [num_kv_heads, seq_len, head_dim] -> GQA expand -> [num_heads, seq_len, head_dim]
        k_t = k_flat.transpose(0, 1).repeat_interleave(groups, dim=0)
        v_t = v_flat.transpose(0, 1).repeat_interleave(groups, dim=0)

        scores = torch.matmul(q_i, k_t.transpose(-2, -1)) * inputs.sm_scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t).squeeze(1)  # [num_heads, head_dim]
        output[bi] = out

    return output.to(torch.bfloat16)


def baseline_prefill_paged(inputs: PrefillPagedInputs) -> torch.Tensor:
    """Naive prefill attention with paged KV cache and causal mask.

    For each batch element, gathers K/V from paged cache, applies causal
    mask, computes attention in float32.
    """
    batch_size = inputs.batch_size
    seq_len = inputs.seq_len
    num_heads = inputs.num_heads
    num_kv_heads = inputs.num_kv_heads
    head_dim = inputs.head_dim
    page_size = inputs.page_size
    groups = num_heads // num_kv_heads

    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_tokens = batch_size * seq_len
    output = torch.empty(total_tokens, num_heads, head_dim, dtype=torch.float32,
                         device=inputs.q.device)

    for bi in range(batch_size):
        q_start = inputs.cu_seqlens_q[bi].item()
        q_end = inputs.cu_seqlens_q[bi + 1].item()
        q_i = inputs.q[q_start:q_end].float()  # [seq_len, num_heads, head_dim]

        blk_ids = inputs.block_ids[bi, :num_pages_per_seq]
        kv_len = inputs.cache_seqlens[bi].item()

        k_flat = inputs.key_cache[blk_ids].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()
        v_flat = inputs.value_cache[blk_ids].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()

        # [num_heads, seq_q, head_dim] and [num_heads, kv_len, head_dim]
        q_t = q_i.transpose(0, 1)
        k_t = k_flat.transpose(0, 1).repeat_interleave(groups, dim=0)
        v_t = v_flat.transpose(0, 1).repeat_interleave(groups, dim=0)

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * inputs.sm_scale

        seq_q = q_t.shape[1]
        causal_mask = torch.tril(
            torch.ones(kv_len, kv_len, device=q_t.device, dtype=torch.bool)
        )[(kv_len - seq_q):, :]
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t).transpose(0, 1)  # [seq_q, num_heads, head_dim]
        output[q_start:q_end] = out

    return output.to(torch.bfloat16)


def baseline_prefill_ragged(inputs: PrefillRaggedInputs) -> torch.Tensor:
    """Naive variable-length prefill attention with continuous Q/K/V and causal mask.

    Iterates per batch element using cu_seqlens boundaries. Computes in float32.
    """
    batch_size = inputs.batch_size
    num_heads = inputs.num_heads
    num_kv_heads = inputs.num_kv_heads
    head_dim = inputs.head_dim
    groups = num_heads // num_kv_heads

    outputs = []
    for bi in range(batch_size):
        q_start = inputs.cu_seqlens_q[bi].item()
        q_end = inputs.cu_seqlens_q[bi + 1].item()
        k_start = inputs.cu_seqlens_kv[bi].item()
        k_end = inputs.cu_seqlens_kv[bi + 1].item()

        qi = inputs.q[q_start:q_end].float()   # [seq_q, num_heads, head_dim]
        ki = inputs.k[k_start:k_end].float()   # [seq_k, num_kv_heads, head_dim]
        vi = inputs.v[k_start:k_end].float()

        ki = ki.transpose(0, 1).repeat_interleave(groups, dim=0)  # [num_heads, seq_k, head_dim]
        vi = vi.transpose(0, 1).repeat_interleave(groups, dim=0)

        scores = torch.einsum("qhd,hkd->hqk", qi, ki) * inputs.sm_scale

        seq_q, seq_k = qi.shape[0], ki.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, device=qi.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("hqk,hkd->qhd", attn, vi)
        outputs.append(out)

    return torch.cat(outputs, dim=0).to(torch.bfloat16)
