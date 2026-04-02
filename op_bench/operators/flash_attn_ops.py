import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache as flash_attn_with_kvcache_fa3

from op_bench.registry import register_operator
from op_bench.inputs import DecodeInputs, PrefillPagedInputs


@register_operator(name="sgl-flash_attn", op_type="decode")
def flash_attn_decode(inputs: DecodeInputs) -> torch.Tensor:
    return flash_attn_with_kvcache_fa3(
        q=inputs.q_fa3,
        k_cache=inputs.key_cache,
        v_cache=inputs.value_cache,
        page_table=inputs.block_ids,
        cache_seqlens=inputs.cache_seqlens,
        causal=True,
    )


@register_operator(name="sgl-flash_attn", op_type="prefill_paged")
def flash_attn_prefill_paged(inputs: PrefillPagedInputs) -> torch.Tensor:
    return flash_attn_with_kvcache_fa3(
        q=inputs.q,
        k_cache=inputs.key_cache,
        v_cache=inputs.value_cache,
        page_table=inputs.block_ids,
        cache_seqlens=inputs.cache_seqlens,
        cu_seqlens_q=inputs.cu_seqlens_q,
        max_seqlen_q=inputs.seq_len,
        softmax_scale=inputs.sm_scale,
        causal=True,
        window_size=(-1, -1),
    )
