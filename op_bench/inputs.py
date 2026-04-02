import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DecodeInputs:
    """Standardized inputs for decode attention (q_seq_len=1, paged KV cache)."""

    # [batch_size, num_heads, head_dim] for HPC/FlashInfer
    q: torch.Tensor
    # [batch_size, 1, num_heads, head_dim] for Flash Attention 3
    q_fa3: torch.Tensor
    # [num_pages, page_size, num_kv_heads, head_dim]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    # [num_pages, 2, page_size, num_kv_heads, head_dim] for FlashInfer
    kv_cache_combined: torch.Tensor
    # [batch_size, max_num_pages_per_seq]
    block_ids: torch.Tensor
    # [batch_size]
    cache_seqlens: torch.Tensor
    # [batch_size + 1], cumulative sequence lengths for queries (for FlashInfer)
    cu_seqlens_q: torch.Tensor   
    # FlashInfer-specific index tensors
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor

    sm_scale: float
    batch_size: int
    seq_len: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int


@dataclass
class PrefillPagedInputs:
    """Standardized inputs for prefill attention with paged KV cache."""

    # [total_tokens, num_heads, head_dim]
    q: torch.Tensor
    # [num_pages, page_size, num_kv_heads, head_dim]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    # [num_pages, 2, page_size, num_kv_heads, head_dim] for FlashInfer
    kv_cache_combined: torch.Tensor
    # [batch_size, max_num_pages_per_seq]
    block_ids: torch.Tensor
    # [batch_size + 1]
    cu_seqlens_q: torch.Tensor
    # [batch_size]
    cache_seqlens: torch.Tensor
    # FlashInfer-specific
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor

    sm_scale: float
    batch_size: int
    seq_len: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int


@dataclass
class PrefillRaggedInputs:
    """Standardized inputs for prefill attention with ragged (continuous) KV."""

    # [total_tokens, num_heads, head_dim]
    q: torch.Tensor
    # [total_tokens, num_kv_heads, head_dim]
    k: torch.Tensor
    v: torch.Tensor
    # [batch_size + 1]
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    # [batch_size]
    seqlens_q: torch.Tensor

    sm_scale: float
    batch_size: int
    seq_len: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

# 顺序
# def _build_block_ids(batch_size: int, num_pages_per_seq: int, device: str) -> torch.Tensor:
#     block_ids = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device=device)
#     for i in range(batch_size):
#         start = i * num_pages_per_seq
#         block_ids[i] = torch.arange(start, start + num_pages_per_seq, dtype=torch.int32, device=device)
#     return block_ids

# 打乱每个序列的页顺序，模拟非连续物理页，但每个批次行的页集相同。当为True时，``kv_indices``必须从``block_ids``填充（与FlashInfer期望的顺序相同）。
def _build_block_ids(
    batch_size: int,
    num_pages_per_seq: int,
    device: str,
    shuffle_pages: bool = True,
) -> torch.Tensor:
    """Assign each sequence a disjoint set of page indices; optionally shuffle order per sequence.

    Shuffling simulates non-contiguous physical pages while keeping the same page set per batch row.
    When True, ``kv_indices`` must be filled from ``block_ids`` (same order as FlashInfer expects).
    """
    block_ids = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        start = i * num_pages_per_seq
        base = torch.arange(start, start + num_pages_per_seq, dtype=torch.int32, device=device)
        if shuffle_pages and num_pages_per_seq > 1:
            perm = torch.randperm(num_pages_per_seq, device=device)
            block_ids[i] = base[perm]
        else:
            block_ids[i] = base
    return block_ids


def generate_decode_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    device: str = "cuda",
    shuffle_pages: bool = True,
) -> DecodeInputs:
    # 参与 decode attention 的页数（与 baseline 使用的 block_ids[:, :n] 一致）
    num_pages_attn = (seq_len + page_size - 1) // page_size
    # 为 HPC new_kv_included=True 多留 1 个物理 page（新 token 写在逻辑位置 seq_len）
    num_pages_per_seq = num_pages_attn + 1
    num_pages = batch_size * num_pages_per_seq + 2

    # 和 hpc-ops pytest 一致：q 和 k 需要除以 sqrt(dim) 进行缩放
    q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device) / math.sqrt(head_dim)
    q_fa3 = q.unsqueeze(1)  # [batch_size, 1, num_heads, head_dim]

    # 生成新的 K/V（对应 q_seq_len=1 的新 token）
    k_new = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) / math.sqrt(head_dim)
    v_new = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    key_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    value_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    block_ids = _build_block_ids(batch_size, num_pages_per_seq, device, shuffle_pages=shuffle_pages)
    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    # 把新的 K/V 写入 KV cache（模拟 new_kv_included=True）
    for i in range(batch_size):
        # 新 token 在 seq_len 位置（0-indexed 的最后一个位置）
        new_token_pos = seq_len  # 因为原有 cache 长度是 seq_len，新 token 在位置 seq_len
        page_id = new_token_pos // page_size
        slot_id = new_token_pos % page_size
        physical_page = block_ids[i, page_id]
        key_cache[physical_page, slot_id] = k_new[i]
        value_cache[physical_page, slot_id] = v_new[i]

    kv_cache_combined = torch.stack([key_cache, value_cache], dim=1)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)

    # FlashInfer 用 (P-1)*page_size + kv_last_page_len 推断 KV 总长；P 必须只含「历史
    # seq_len 个 token」所占的页，不能把 new_kv 多出来的那页放进 kv_indices，否则会多读 KV。
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages_attn
    kv_indices = torch.zeros(batch_size * num_pages_attn, dtype=torch.int32, device=device)
    for i in range(batch_size):
        kv_indices[i * num_pages_attn : (i + 1) * num_pages_attn] = block_ids[i, :num_pages_attn]

    last_page_len = seq_len % page_size if seq_len % page_size != 0 else page_size
    kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device=device)

    return DecodeInputs(
        q=q,
        q_fa3=q_fa3,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_cache_combined=kv_cache_combined,
        block_ids=block_ids,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        sm_scale=1.0 / math.sqrt(head_dim),
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )


def generate_prefill_paged_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    device: str = "cuda",
    shuffle_pages: bool = True,
) -> PrefillPagedInputs:
    total_tokens = batch_size * seq_len
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_seq + 2

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)

    kv_cache_combined = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device=device,
    )
    key_cache = kv_cache_combined[:, 0, :, :, :].contiguous()
    value_cache = kv_cache_combined[:, 1, :, :, :].contiguous()

    block_ids = _build_block_ids(batch_size, num_pages_per_seq, device, shuffle_pages=shuffle_pages)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    qo_indptr = cu_seqlens_q.clone()
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages_per_seq

    kv_indices = torch.zeros(batch_size * num_pages_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        kv_indices[i * num_pages_per_seq : (i + 1) * num_pages_per_seq] = block_ids[i]

    last_page_len = seq_len % page_size if seq_len % page_size != 0 else page_size
    kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device=device)

    return PrefillPagedInputs(
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_cache_combined=kv_cache_combined,
        block_ids=block_ids,
        cu_seqlens_q=cu_seqlens_q,
        cache_seqlens=cache_seqlens,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        sm_scale=1.0 / math.sqrt(head_dim),
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )


def generate_prefill_ragged_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str = "cuda",
) -> PrefillRaggedInputs:
    total_tokens = batch_size * seq_len

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    cu_seqlens_kv = cu_seqlens_q.clone()
    seqlens_q = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    return PrefillRaggedInputs(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqlens_q=seqlens_q,
        sm_scale=1.0 / math.sqrt(head_dim),
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
