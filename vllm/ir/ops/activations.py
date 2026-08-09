# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def silu_and_mul(x: Tensor) -> Tensor:
    """SwiGLU: silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2"""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


@silu_and_mul.register_input_generator
def _silu_and_mul_inputs(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> tuple:
    x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype, device=device)
    return (x,)
