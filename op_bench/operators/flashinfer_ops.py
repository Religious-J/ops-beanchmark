import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from op_bench.registry import register_operator
from op_bench.inputs import DecodeInputs, PrefillPagedInputs, PrefillRaggedInputs

_WORKSPACE_SIZE = 128 * 1024 * 1024


# ---- decode ----
def _decode_setup(inputs: DecodeInputs):
    workspace = torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=inputs.q.device)
    # use_tensor_cores=True 在 decode 场景下（q_seq_len=1）性能下降。decode 时 Q 只有一行，矩阵太小不适合 tensor core
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=False)
    wrapper.begin_forward(
        inputs.kv_indptr,
        inputs.kv_indices,
        inputs.kv_last_page_len,
        inputs.num_heads,
        inputs.num_kv_heads,
        inputs.head_dim,
        inputs.page_size,
        q_data_type=torch.bfloat16,
    )
    inputs._fi_workspace = workspace
    inputs._fi_wrapper = wrapper


@register_operator(name="flashinfer", op_type="decode", setup=_decode_setup)
def flashinfer_decode(inputs: DecodeInputs) -> torch.Tensor:
    return inputs._fi_wrapper.forward(inputs.q, inputs.kv_cache_combined, sm_scale=inputs.sm_scale)


# ---- prefill_paged ----
def _prefill_paged_setup(inputs: PrefillPagedInputs):
    workspace = torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=inputs.q.device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.begin_forward(
        inputs.qo_indptr,
        inputs.kv_indptr,
        inputs.kv_indices,
        inputs.kv_last_page_len,
        inputs.num_heads,
        inputs.num_kv_heads,
        inputs.head_dim,
        inputs.page_size,
        q_data_type=torch.bfloat16,
    )
    inputs._fi_workspace = workspace
    inputs._fi_wrapper = wrapper


@register_operator(name="flashinfer", op_type="prefill_paged", setup=_prefill_paged_setup)
def flashinfer_prefill_paged(inputs: PrefillPagedInputs) -> torch.Tensor:
    return inputs._fi_wrapper.forward(inputs.q, inputs.kv_cache_combined, causal=True, sm_scale=inputs.sm_scale)


# ---- prefill_ragged ----
def _prefill_ragged_setup(inputs: PrefillRaggedInputs):
    workspace = torch.empty(_WORKSPACE_SIZE, dtype=torch.uint8, device=inputs.q.device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    wrapper.begin_forward(
        inputs.cu_seqlens_q,
        inputs.cu_seqlens_kv,
        inputs.num_heads,
        inputs.num_kv_heads,
        inputs.head_dim,
        q_data_type=torch.bfloat16,
    )
    inputs._fi_workspace = workspace
    inputs._fi_wrapper = wrapper


@register_operator(name="flashinfer", op_type="prefill_ragged", setup=_prefill_ragged_setup)
def flashinfer_prefill_ragged(inputs: PrefillRaggedInputs) -> torch.Tensor:
    return inputs._fi_wrapper.forward(inputs.q, inputs.k, inputs.v, causal=True, sm_scale=inputs.sm_scale)
