import torch
import hpc

from op_bench.registry import register_operator
from op_bench.inputs import DecodeInputs, PrefillPagedInputs, PrefillRaggedInputs


@register_operator(name="hpc", op_type="decode")
def hpc_decode(inputs: DecodeInputs) -> torch.Tensor:
    return hpc.attention_decode_bf16(
        q=inputs.q.reshape(-1, inputs.num_heads, inputs.head_dim),
        kcache=inputs.key_cache,
        vcache=inputs.value_cache,
        block_ids=inputs.block_ids,
        num_seq_kvcache=inputs.cache_seqlens,
        new_kv_included=True,
        splitk=False,
    )


@register_operator(name="hpc_splitk", op_type="decode")
def hpc_splitk_decode(inputs: DecodeInputs) -> torch.Tensor:
    return hpc.attention_decode_bf16(
        q=inputs.q.reshape(-1, inputs.num_heads, inputs.head_dim),
        kcache=inputs.key_cache,
        vcache=inputs.value_cache,
        block_ids=inputs.block_ids,
        num_seq_kvcache=inputs.cache_seqlens,
        new_kv_included=True,
        splitk=True,
    )


@register_operator(name="hpc", op_type="prefill_paged")
def hpc_prefill_paged(inputs: PrefillPagedInputs) -> torch.Tensor:
    return hpc.attention_with_kvcache_prefill_bf16(
        q=inputs.q,
        kcache=inputs.key_cache,
        vcache=inputs.value_cache,
        cu_seqlens_q=inputs.cu_seqlens_q,
        block_ids=inputs.block_ids,
        seqlens_kvcache=inputs.cache_seqlens,
        max_seqlens_q=inputs.seq_len,
    )


@register_operator(name="hpc", op_type="prefill_ragged")
def hpc_prefill_ragged(inputs: PrefillRaggedInputs) -> torch.Tensor:
    return hpc.attention_prefill_bf16(
        q=inputs.q,
        k=inputs.k,
        v=inputs.v,
        seqlens_q=inputs.seqlens_q,
        cu_seqlens_q=inputs.cu_seqlens_q,
        max_seqlens_q=inputs.seq_len,
    )
