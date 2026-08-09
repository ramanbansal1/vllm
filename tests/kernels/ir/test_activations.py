# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

# This registers op implementations
import vllm.kernels  # noqa: F401
from tests.ir.ir_test_utils import (
    COMMON_HIDDEN_SIZES,
    NUM_TOKENS,
    assert_close,
    clone_args,
    supported_providers,
)
from vllm import ir
from vllm.platforms import current_platform

silu_and_mul_native = ir.ops.silu_and_mul.impls["native"].impl_fn

IS_GPGPU_DEVICE = current_platform.is_cuda_alike() or current_platform.is_xpu()


# ── 1. Registration test ──────────────────────────────────────────────

@pytest.mark.skipif(
    not IS_GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
def test_silu_and_mul_registration():
    expected = {
        "native": True,
        "vllm_c": IS_GPGPU_DEVICE,
    }

    actual = {
        provider: impl.supported
        for provider, impl in ir.ops.silu_and_mul.impls.items()
    }

    assert actual == expected


# ── 2 & 3. Parameterized test class ───────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", COMMON_HIDDEN_SIZES)
@pytest.mark.skipif(
    not IS_GPGPU_DEVICE,
    reason="Currently only kernels on CUDA, ROCm and XPU",
)
class TestSiluAndMul:
    def test_native_semantics(self, dtype, n_tokens, hidden_size):
        """Validate shape/dtype contracts and mathematical properties."""
        (x,) = ir.ops.silu_and_mul.generate_inputs(
            num_tokens=n_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=current_platform.device_type,
        )

        out = silu_and_mul_native(x)

        # Output shape should be half the last dimension
        d = x.shape[-1] // 2
        expected_shape = x.shape[:-1] + (d,)
        assert out.shape == expected_shape
        assert out.dtype == x.dtype
        assert out.device == x.device

        # Verify against explicit silu computation
        gate = x[..., :d]
        up = x[..., d:]
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.silu_and_mul)
    )
    def test_impls(self, dtype, n_tokens, hidden_size, provider):
        """Compare provider output against native reference."""
        impl = ir.ops.silu_and_mul.impls[provider]
        (x,) = ir.ops.silu_and_mul.generate_inputs(
            num_tokens=n_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=current_platform.device_type,
        )
        args = (x,)

        if not impl.supports_args(*args):
            pytest.skip(f"{provider} does not support args")

        ref_output = silu_and_mul_native(*clone_args(args))
        output = impl.impl_fn(*clone_args(args))
        assert_close(ir.ops.silu_and_mul, output, ref_output)

        # Check that dispatched call matches direct call
        with ir.ops.silu_and_mul.set_priority([provider, "native"]):
            out_dispatched = ir.ops.silu_and_mul(*args)
        out_direct = impl.impl_fn(*args)
        torch.testing.assert_close(
            out_dispatched, out_direct, rtol=0.0, atol=0.0
        )

    @pytest.mark.parametrize(
        "provider", supported_providers(ir.ops.silu_and_mul)
    )
    def test_torch_opcheck(self, dtype, n_tokens, hidden_size, provider):
        """Validate torch.library op contract (autograd, fake tensor)."""
        args = ir.ops.silu_and_mul.generate_inputs(
            num_tokens=n_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=current_platform.device_type,
        )

        with ir.ops.silu_and_mul.set_priority([provider, "native"]):
            torch.library.opcheck(
                torch.ops.vllm_ir.silu_and_mul, args
            )
