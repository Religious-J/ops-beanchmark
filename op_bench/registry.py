from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class OpType(str, Enum):
    DECODE = "decode"
    PREFILL_PAGED = "prefill_paged"
    PREFILL_RAGGED = "prefill_ragged"


@dataclass
class RegisteredOp:
    """An operator with optional setup/teardown phases separated from forward.

    - forward(inputs) -> Tensor: the kernel call being timed.
    - setup(inputs) -> None: optional prep work (e.g. FlashInfer begin_forward)
      executed once before warmup+timing loops, outside the timer.
    - teardown(inputs) -> None: optional cleanup after timing.
    """
    forward: Callable
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None


_REGISTRY: Dict[OpType, Dict[str, RegisteredOp]] = {op_type: {} for op_type in OpType}


def register_operator(
    name: str,
    op_type: str,
    setup: Optional[Callable] = None,
    teardown: Optional[Callable] = None,
):
    """Decorator to register an attention operator implementation.

    Usage (simple, no setup needed):
        @register_operator(name="hpc", op_type="decode")
        def hpc_decode(inputs): ...

    Usage (with setup phase, e.g. FlashInfer):
        def flashinfer_decode_setup(inputs):
            # allocate workspace, create wrapper, begin_forward
            inputs._flashinfer_wrapper = wrapper

        @register_operator(name="flashinfer", op_type="decode",
                           setup=flashinfer_decode_setup)
        def flashinfer_decode(inputs):
            return inputs._flashinfer_wrapper.forward(...)
    """
    op_type_enum = OpType(op_type)

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY[op_type_enum]:
            raise ValueError(
                f"Operator '{name}' already registered for op_type '{op_type}'"
            )
        _REGISTRY[op_type_enum][name] = RegisteredOp(
            forward=fn, setup=setup, teardown=teardown,
        )
        return fn

    return decorator


def get_registry() -> Dict[OpType, Dict[str, RegisteredOp]]:
    return _REGISTRY
