import time
import math
import torch
import hpc

from sgl_kernel.flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_fa3
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

def test_decode_accuracy_flash_infer(
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    device="cuda"
):
    q_seq_len = 1     # Decode benchmark only supports seq_len=1

    q = torch.randn(
         batch_size * q_seq_len, num_heads, head_dim,
         dtype=torch.bfloat16, device=device
        )

    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * (seq_len + page_size - 1) // page_size + 2
    key_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    value_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    # ========== FlashInfer 计算 ==========
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=True)

    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages_per_seq
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device=device)

    last_page_len = seq_len % page_size if seq_len % page_size != 0 else page_size
    for i in range(batch_size):
        kv_last_page_len[i] = last_page_len
    
    kv_cache_flash = torch.stack([key_cache, value_cache], dim=1)

    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
    )

    sm_scale = 1.0 / math.sqrt(head_dim)
    output_flash = wrapper.forward(q, kv_cache_flash, sm_scale=sm_scale)

    # ========== HPC 计算 ==========
    max_num_pages_per_seq = (seq_len + page_size - 1) // page_size
    block_ids = torch.zeros(batch_size, max_num_pages_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        start_page = i * max_num_pages_per_seq
        page_indices = torch.arange(start_page, start_page + max_num_pages_per_seq, dtype=torch.int32, device=device)
        block_ids[i] = page_indices

    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    output_hpc = hpc.attention_decode_bf16(
        q=q.reshape(-1, num_heads, head_dim),
        kcache=key_cache,
        vcache=value_cache,
        block_ids=block_ids,
        num_seq_kvcache=cache_seqlens,
        new_kv_included=True,
        splitk=True,
    )

    assert torch.allclose(output_flash.view(-1), output_hpc.view(-1), atol=0.016, )
    print("decode accuracy PASS!!!")


def test_prefill_accuracy_flashinfer_gqa(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    device="cuda"
):
    """Test FlashInfer BatchPrefillWithPagedKVCacheWrapper GQA against HPC attention_with_kvcache_prefill_bf16"""
    total_tokens = batch_size * seq_len

    q = torch.randn(total_tokens, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device)

    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_seq + 2

    # KV cache for FlashInfer: [num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_cache_flash = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device=device
    )

    # KV cache for HPC: [num_pages, page_size, num_kv_heads, head_dim]
    key_cache_hpc = kv_cache_flash[:, 0, :, :, :].contiguous()
    value_cache_hpc = kv_cache_flash[:, 1, :, :, :].contiguous()

    # ========== FlashInfer 计算 ==========
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

    # Prepare index tensors
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages_per_seq

    # Create kv_indices (page indices for each sequence)
    kv_indices = torch.zeros(batch_size * num_pages_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        start_page = i * num_pages_per_seq
        kv_indices[i * num_pages_per_seq:(i + 1) * num_pages_per_seq] = torch.arange(
            start_page, start_page + num_pages_per_seq, dtype=torch.int32, device=device
        )

    # Last page length
    last_page_len = seq_len % page_size if seq_len % page_size != 0 else page_size
    kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device=device)

    # Begin forward
    wrapper.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
    )

    # Compute attention
    sm_scale = 1.0 / math.sqrt(head_dim)
    output_flash = wrapper.forward(
        q, kv_cache_flash, causal=True, sm_scale=sm_scale
    )

    # ========== HPC 计算 ==========
    # Prepare block_ids for HPC
    block_ids = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        start_page = i * num_pages_per_seq
        page_indices = torch.arange(start_page, start_page + num_pages_per_seq, dtype=torch.int32, device=device)
        block_ids[i] = page_indices

    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len

    output_hpc = hpc.attention_with_kvcache_prefill_bf16(
        q=q,
        kcache=key_cache_hpc,
        vcache=value_cache_hpc,
        cu_seqlens_q=cu_seqlens_q,
        block_ids=block_ids,
        seqlens_kvcache=cache_seqlens,
        max_seqlens_q=seq_len,
    )

    assert torch.allclose(output_flash, output_hpc, atol=0.016), \
        f"Max diff: {(output_flash - output_hpc).abs().max().item()}"
    print("decode PASS!!!")


## ----- decode -----
def benchmark_attention_decode_bf16(
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    new_kv_included=True,
    device = "cuda"
):
    q_seq_len = 1     # Decode benchmark only supports seq_len=1
    q = torch.randn(
         batch_size, q_seq_len, num_heads, head_dim,
         dtype=torch.bfloat16, device=device
        )

    num_pages = batch_size * (seq_len + int(new_kv_included) + page_size - 1) // page_size + 2
    key_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    value_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    max_num_pages_per_seq = (seq_len + int(new_kv_included) + page_size - 1) // page_size
    block_ids = torch.zeros(batch_size, max_num_pages_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        start_page = i * max_num_pages_per_seq
        page_indices = torch.arange(start_page, start_page + max_num_pages_per_seq, dtype=torch.int32, device=device)
        block_ids[i] = page_indices

    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    output_flash = flash_attn_with_kvcache_fa3(
        q=q,
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=block_ids,
        cache_seqlens=cache_seqlens,
        causal=True,
    )

    torch.cuda.synchronize()
    flash_time = time.perf_counter() - start_time

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    output_hpc = hpc.attention_decode_bf16(
        q=q.reshape(-1, num_heads, head_dim),
        kcache=key_cache,
        vcache=value_cache,
        block_ids=block_ids,
        num_seq_kvcache=cache_seqlens,
        new_kv_included=new_kv_included,
        splitk=True,
    )

    torch.cuda.synchronize()
    hpc_time = time.perf_counter() - start_time

    assert torch.allclose(output_flash.view(-1), output_hpc.view(-1), atol=0.016, )
    print("decode PASS!!!")

## ----- prefill -----
def benchmark_attention_with_kvcache_prefill_bf16(
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    device = "cuda"
):
    total_tokens = batch_size * seq_len

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    num_pages = (total_tokens + page_size - 1) // page_size + 2
    key_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    value_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    max_num_pages_per_seq = (seq_len + page_size - 1) // page_size
    block_ids = torch.zeros(batch_size, max_num_pages_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        start_page = i * max_num_pages_per_seq
        page_indices = torch.arange(start_page, start_page + max_num_pages_per_seq, dtype=torch.int32, device=device)
        block_ids[i] = page_indices

    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    output_flash = flash_attn_with_kvcache_fa3(
        q=q,
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=block_ids,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=seq_len,
        softmax_scale=1.0 / (head_dim ** 0.5),
        causal=True,
        window_size=(-1, -1),
    )

    torch.cuda.synchronize()
    flash_time = time.perf_counter() - start_time

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    output_hpc = hpc.attention_with_kvcache_prefill_bf16(
        q=q,
        kcache=key_cache,
        vcache=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        block_ids=block_ids,
        seqlens_kvcache=cache_seqlens,
        max_seqlens_q=seq_len,
    )

    torch.cuda.synchronize()
    hpc_time = time.perf_counter() - start_time

    assert torch.allclose(output_flash, output_hpc, atol=0.016, )
    print("prefill PASS!!!")

## ------------------- 内部测试 -------------------
def convert_to_paged_format(kv, batch_size, seq_len, page_size, num_kv_heads, head_dim):
    """
    Convert KV tensor from [total_seq, num_kv_heads, head_dim] to paged format
    [num_blocks, page_size, num_kv_heads, head_dim]
    """
    total_tokens = batch_size * seq_len
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_seq

    # Reshape to [batch_size, seq_len, num_kv_heads, head_dim]
    kv_reshaped = kv.view(batch_size, seq_len, num_kv_heads, head_dim)

    # Pad seq_len to multiple of page_size
    padded_seq_len = num_pages_per_seq * page_size
    if padded_seq_len > seq_len:
        padding = torch.zeros(
            batch_size, padded_seq_len - seq_len, num_kv_heads, head_dim,
            dtype=kv.dtype, device=kv.device
        )
        kv_padded = torch.cat([kv_reshaped, padding], dim=1)
    else:
        kv_padded = kv_reshaped

    # Reshape to [batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim]
    kv_paged = kv_padded.view(batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim)

    # Merge batch and page dimensions: [num_pages, page_size, num_kv_heads, head_dim]
    kv_paged = kv_paged.view(num_pages, page_size, num_kv_heads, head_dim)

    return kv_paged.contiguous(), num_pages_per_seq


def test_hpc_prefill_vs_kvcache_prefill(
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    head_dim,
    page_size,
    device="cuda"
):
    """
    Compare hpc.attention_prefill_bf16 (direct Q/K/V) with
    hpc.attention_with_kvcache_prefill_bf16 (paged KV cache)
    """
    print(f"Testing batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, "
          f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}")

    total_tokens = batch_size * seq_len

    # 1. Generate identical Q/K/V data in BF16
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    # 2. Prepare index tensors for direct prefill
    seqlens_q = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len

    # 3. Convert K/V to paged format for kvcache version
    k_cache, num_pages_per_seq = convert_to_paged_format(
        k, batch_size, seq_len, page_size, num_kv_heads, head_dim
    )
    v_cache, _ = convert_to_paged_format(
        v, batch_size, seq_len, page_size, num_kv_heads, head_dim
    )

    # 4. Create block_ids for each sequence
    block_ids = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device=device)
    for i in range(batch_size):
        start_page = i * num_pages_per_seq
        page_indices = torch.arange(start_page, start_page + num_pages_per_seq,
                                    dtype=torch.int32, device=device)
        block_ids[i] = page_indices

    # 5. seqlens_kvcache is the same as seqlens_q
    seqlens_kvcache = seqlens_q.clone()

    # ========== Direct Prefill (attention_prefill_bf16) ==========
    output_direct = hpc.attention_prefill_bf16(
        q=q,
        k=k,
        v=v,
        seqlens_q=seqlens_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlens_q=seq_len,
    )

    # ========== Paged KV Cache Prefill (attention_with_kvcache_prefill_bf16) ==========
    output_kvcache = hpc.attention_with_kvcache_prefill_bf16(
        q=q,
        kcache=k_cache,
        vcache=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        block_ids=block_ids,
        seqlens_kvcache=seqlens_kvcache,
        max_seqlens_q=seq_len,
    )

    assert torch.allclose(output_direct, output_kvcache, atol=0.016)
    print("  ✓ PASS: Direct prefill matches paged KV-cache prefill")


def test_prefill_accuracy_flashinfer_ragged_gqa(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    device="cuda"
):
    """Test FlashInfer BatchPrefillWithRaggedKVCacheWrapper GQA against HPC attention_prefill_bf16"""

    total_tokens = batch_size * seq_len

    # Generate identical Q/K/V data in BF16
    q = torch.randn(total_tokens, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    # ========== FlashInfer Ragged KV Cache computation ==========
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, "NHD")

    # Prepare cumulative sequence lengths
    cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    cu_seqlens_kv = cu_seqlens_q.clone()

    # Begin forward
    wrapper.begin_forward(
        cu_seqlens_q,
        cu_seqlens_kv,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=torch.bfloat16,
    )

    # Compute attention
    sm_scale = 1.0 / math.sqrt(head_dim)
    output_flash = wrapper.forward(
        q, k, v, causal=True, sm_scale=sm_scale
    )

    # ========== HPC computation ==========
    # Prepare seqlens for HPC
    seqlens_q = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    output_hpc = hpc.attention_prefill_bf16(
        q=q,
        k=k,
        v=v,
        seqlens_q=seqlens_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlens_q=seq_len,
    )

    # Compare results
    assert torch.allclose(output_flash, output_hpc, atol=0.016)
    print(f"✓ Ragged GQA test PASS!")

## ----------------------------------------

if __name__ == "__main__":
    num_batchs = [1, 2, 8, 16, 32]
    seq_lens = [128, 512, 1024, 4096]
    block_size = 64
    num_head_qs = [4, 8]
    num_head_kv = 1
    dim = 128
    for batch_size in num_batchs:
        for seq_len in seq_lens:
            for num_head_q in num_head_qs:
                print(f"batch_size={batch_size}, seq_len={seq_len}, num_head_q={num_head_q}, num_head_kv={num_head_kv}, dim={dim}, block_size={block_size}")

                test_decode_accuracy_flash_infer(
                    batch_size,
                    seq_len,
                    num_head_q,
                    num_head_kv,
                    dim,
                    block_size,
                )

                test_prefill_accuracy_flashinfer_gqa(
                    batch_size,
                    seq_len,
                    num_head_q,
                    num_head_kv,
                    dim,
                    block_size,
                )

                benchmark_attention_with_kvcache_prefill_bf16(
                    batch_size,
                    seq_len,
                    num_head_q,
                    num_head_kv,
                    dim,
                    block_size,
                )

                benchmark_attention_decode_bf16(
                    batch_size,
                    seq_len,
                    num_head_q,
                    num_head_kv,
                    dim,
                    block_size,
                )

                test_hpc_prefill_vs_kvcache_prefill(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_heads=num_head_q,
                    num_kv_heads=num_head_kv,
                    head_dim=dim,
                    page_size=block_size,
                )

                # Test ragged GQA (BatchPrefillWithRaggedKVCacheWrapper)
                test_prefill_accuracy_flashinfer_ragged_gqa(
                    batch_size,
                    seq_len,
                    num_head_q,
                    num_head_kv,
                    dim,
                )
  
