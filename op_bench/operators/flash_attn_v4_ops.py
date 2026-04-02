"""Flash Attention v4 via sgl_kernel ``_fa4_interface.flash_attn_varlen_func``.

We call FA4 **directly** instead of ``flash_attn_with_kvcache(..., ver=4)``, because
some sgl-kernel builds hit ``NameError: cu_seqlens_q is not defined`` inside the
``flash_attn_with_kvcache`` FA4 branch (wrapper bug / version skew).

Requires: ``import sgl_kernel._fa4_interface`` (FA4 + flash_attn_origin / cutlass-dsl).

Decode: Q as ``[B, H, D]`` + ``cu_seqlens_q = [0,1,...,B]``; ``seqused_k`` = cache lengths;
``page_table`` = block ids. See SGLang FA4 MHA notes on ``page_size=128``.
"""

import torch

from op_bench.registry import register_operator
from op_bench.inputs import DecodeInputs, PrefillPagedInputs

try:
    import sgl_kernel._fa4_interface  # noqa: F401
    from sgl_kernel._fa4_interface import flash_attn_varlen_func as _fa4_attn
    _SGL_FA4_AVAILABLE = True
except ImportError:
    _SGL_FA4_AVAILABLE = False
    _fa4_attn = None


def _fa4_out_tensor(out):
    """FA4 returns tensor or (out, lse) when return_softmax_lse=True."""
    if isinstance(out, tuple):
        return out[0]
    return out


if _SGL_FA4_AVAILABLE:

    @register_operator(name="flash_attn_v4", op_type="decode")
    def flash_attn_v4_decode(inputs: DecodeInputs) -> torch.Tensor:
        # q = inputs.q.contiguous()
        out = _fa4_attn(
            inputs.q,
            inputs.key_cache,
            inputs.value_cache,
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=inputs.cache_seqlens,
            page_table=inputs.block_ids,
            softmax_scale=inputs.sm_scale,
            causal=True,
            window_size=(None, None),
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            return_softmax_lse=False,
            score_mod=None,
            aux_tensors=None,
        )
        return _fa4_out_tensor(out)

    @register_operator(name="flash_attn_v4", op_type="prefill_paged")
    def flash_attn_v4_prefill_paged(inputs: PrefillPagedInputs) -> torch.Tensor:
        out = _fa4_attn(
            inputs.q.contiguous(),
            inputs.key_cache,
            inputs.value_cache,
            cu_seqlens_q=inputs.cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=inputs.cache_seqlens,
            page_table=inputs.block_ids,
            softmax_scale=inputs.sm_scale,
            causal=True,
            window_size=(None, None),
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            return_softmax_lse=False,
            score_mod=None,
            aux_tensors=None,
        )
        return _fa4_out_tensor(out)
