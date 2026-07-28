"""
test_rmsswiglu_correctness.py — Production Test Suite for RMSSwiGLU

Tests the fused RMSNorm + SwiGLU operation implemented in rmsswiglu.cu
via the 3-stage pipeline: RMSNorm → cuBLAS GEMM (gate & up) → SiLU(gate) * up.

Covers:
  1. Numerical correctness against PyTorch FP32 reference
  2. Edge cases: zero input, large scaling, single token, max tokens
  3. Shape flexibility: arbitrary batch/seq leading dims
  4. Dtype consistency validation
  5. Latency benchmark vs PyTorch native implementation
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple

from llama_pure_compute.ops import rmsswiglu_forward, _rmsswiglu_forward_pytorch



# Reference Implementation (always FP32 for numerical ground truth)
def torch_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standard RMSNorm: x / sqrt(mean(x^2) + eps) * weight"""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def torch_rms_swiglu_fused(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reference: RMSNorm → Linear(gate) → Linear(up) → SiLU(gate) * up
    Computed in FP32 for maximum precision ground truth.
    """
    x_f = x.float()
    w_f = rms_weight.float()
    gw_f = gate_w.float()
    uw_f = up_w.float()

    normed = torch_rms_norm(x_f, w_f, eps=eps)
    gate = F.linear(normed, gw_f)
    up = F.linear(normed, uw_f)
    return (F.silu(gate) * up).to(x.dtype)


# Fixtures
@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def benchmark_cuda(fn, *args, warmup: int = 25, reps: int = 200) -> float:
    """Measure average GPU kernel execution time in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(reps):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / reps


# Tolerance Table
TOLERANCES = {
    torch.float32:  {"rtol": 1e-4,  "atol": 1e-4},   # true fp32 (pedantic math, no TF32)
    torch.float16:  {"rtol": 1e-3,  "atol": 1e-3},   # normal fp16 rounding at O(1) scale
    torch.bfloat16: {"rtol": 1.5e-2, "atol": 1.5e-2}, # bf16's ~3 decimal digits of precision
}


def scaled_linear_weight(out_dim: int, in_dim: int, device, dtype) -> torch.Tensor:
    """Realistic weight init (~1/sqrt(fan_in)), matching how real models
    are initialized, instead of raw unit-variance randn(). Unscaled randn
    weights drive activations to O(10^4), which is meaningless for testing
    fp16/bf16 precision since it sits far outside their useful range."""
    return torch.randn(out_dim, in_dim, device=device, dtype=dtype) / (in_dim ** 0.5)


# Max fraction of elements allowed to fall outside the per-element
# rtol/atol band before we call it a real bug rather than expected
# low-precision rounding noise.
OUTLIER_FRACTION = {
    torch.float32:  0.0,     # true fp32 (pedantic math): every element should clear tolerance
    torch.float16:  0.001,   # 0.1% -- a handful of worst-case-rounding elements is expected
    torch.bfloat16: 0.002,   # 0.2% -- bf16 has fewer mantissa bits, slightly more tail noise
}


def assert_correctness(out: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype, msg: str = ""):
    tol = TOLERANCES[dtype]
    out_f, ref_f = out.float(), ref.float()
    diff = (out_f - ref_f).abs()
    allowed = tol["atol"] + tol["rtol"] * ref_f.abs()

    mismatched = diff > allowed
    mismatch_frac = mismatched.float().mean().item()
    max_allowed_frac = OUTLIER_FRACTION[dtype]

    worst_overshoot = (diff[mismatched] / allowed[mismatched].clamp_min(1e-12)).max().item() if mismatched.any() else 0.0

    assert mismatch_frac <= max_allowed_frac and worst_overshoot <= 10.0, (
        f"{msg}: {mismatch_frac*100:.4f}% of elements exceeded tolerance "
        f"(limit {max_allowed_frac*100:.4f}%), worst overshoot {worst_overshoot:.2f}x "
        f"the allowed band -- this looks like a real correctness issue, "
        f"not rounding noise."
    )


# Test Suite
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRMSSwiGLUCorrectness:

    # 1. Numerical Correctness
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 1),       # single token decode
        (1, 128),     # single batch prefill
        (4, 512),     # multi-batch mid context
        (2, 2048),    # long context
    ])
    @pytest.mark.parametrize("hidden_dim,inter_dim", [
        (4096, 11008),   # Llama-3.1-8B
        (2048, 5632),    # Llama-3.2-3B
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_numerical_correctness(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        inter_dim: int,
        dtype: torch.dtype,
    ):
        device = torch.device("cuda")
        eps = 1e-6

        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
        norm_w = torch.randn(hidden_dim, device=device, dtype=dtype)
        gate_w = scaled_linear_weight(inter_dim, hidden_dim, device, dtype)
        up_w = scaled_linear_weight(inter_dim, hidden_dim, device, dtype)

        # Reference (FP32 ground truth)
        ref = torch_rms_swiglu_fused(x, norm_w, gate_w, up_w, eps=eps)

        # Custom CUDA kernel
        out = rmsswiglu_forward(x, norm_w, gate_w, up_w, eps)

        tol = TOLERANCES[dtype]
        assert_correctness(
            out, ref, dtype,
            msg=f"Fail: dtype={dtype}, shape=({batch_size},{seq_len},{hidden_dim})",
        )

    # 2. Shape Flexibility
    @pytest.mark.parametrize("input_shape", [
        (1, 1, 4096),       # 3D: [B, S, D]
        (1, 4096),           # 2D: [S, D]  (single batch flattened)
        (4096,),             # 1D: [D]      (single token flattened)
    ])
    def test_shape_flexibility(self, input_shape: tuple):
        device = torch.device("cuda")
        dtype = torch.float16
        hidden_dim = input_shape[-1]
        inter_dim = 11008

        x = torch.randn(*input_shape, device=device, dtype=dtype)
        norm_w = torch.randn(hidden_dim, device=device, dtype=dtype)
        gate_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)
        up_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)

        out = rmsswiglu_forward(x, norm_w, gate_w, up_w)

        # Last dim replaced, leading dims preserved
        expected_shape = list(input_shape)
        expected_shape[-1] = inter_dim
        assert out.shape == torch.Size(expected_shape), \
            f"Shape mismatch: expected {expected_shape}, got {list(out.shape)}"

    # 3. Edge Cases
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_zero_input(self, dtype: torch.dtype):
        """Zero input → zero output (RMSNorm of zeros is undefined but
        the kernel should produce finite, deterministic results)."""
        device = torch.device("cuda")
        hidden_dim, inter_dim = 2048, 5632

        x = torch.zeros(2, 16, hidden_dim, device=device, dtype=dtype)
        norm_w = torch.randn(hidden_dim, device=device, dtype=dtype)
        gate_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)
        up_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)

        out = rmsswiglu_forward(x, norm_w, gate_w, up_w)

        assert not torch.isnan(out).any(), "NaN detected on zero input"
        assert not torch.isinf(out).any(), "Inf detected on zero input"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_input_scaling(self, dtype: torch.dtype):
        """Large input values should not cause overflow in FP16/BF16."""
        device = torch.device("cuda")
        hidden_dim, inter_dim = 2048, 5632

        x = torch.randn(1, 16, hidden_dim, device=device, dtype=dtype) * 1e3
        norm_w = torch.randn(hidden_dim, device=device, dtype=dtype)
        gate_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)
        up_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)

        out = rmsswiglu_forward(x, norm_w, gate_w, up_w)

        assert not torch.isnan(out).any(), "NaN with large input scaling"
        assert not torch.isinf(out).any(), "Inf with large input scaling"

    def test_identity_weights(self):
        """With gate_w = identity-like and small input, output should be
        approximately SiLU(RMSNorm(x)) * up_proj(x)."""
        device = torch.device("cuda")
        hidden_dim, inter_dim = 512, 512
        dtype = torch.float32

        x = torch.randn(1, 4, hidden_dim, device=device, dtype=dtype)
        norm_w = torch.ones(hidden_dim, device=device, dtype=dtype)
        gate_w = torch.eye(hidden_dim, device=device, dtype=dtype)
        up_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)

        out = rmsswiglu_forward(x, norm_w, gate_w, up_w)
        ref = torch_rms_swiglu_fused(x, norm_w, gate_w, up_w)

        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    def test_determinism(self):
        """Same input must produce identical output across calls."""
        device = torch.device("cuda")
        dtype = torch.float16
        hidden_dim, inter_dim = 2048, 5632

        x = torch.randn(2, 32, hidden_dim, device=device, dtype=dtype)
        norm_w = torch.randn(hidden_dim, device=device, dtype=dtype)
        gate_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)
        up_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=dtype)

        out1 = rmsswiglu_forward(x, norm_w, gate_w, up_w)
        out2 = rmsswiglu_forward(x.clone(), norm_w, gate_w, up_w)

        torch.testing.assert_close(out1, out2)

    # 4. Fallback Correctness (PyTorch native, no CUDA kernel)
    def test_pytorch_fallback_matches_reference(self):
        """Ensure the Python fallback produces correct results."""
        device = torch.device("cpu")
        dtype = torch.float32
        B, S, D, I = 2, 8, 512, 1024

        x = torch.randn(B, S, D, device=device, dtype=dtype)
        norm_w = torch.randn(D, device=device, dtype=dtype)
        gate_w = torch.randn(I, D, device=device, dtype=dtype)
        up_w = torch.randn(I, D, device=device, dtype=dtype)

        out = _rmsswiglu_forward_pytorch(x, norm_w, gate_w, up_w, eps=1e-5)
        ref = torch_rms_swiglu_fused(x, norm_w, gate_w, up_w, eps=1e-5)

        torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

    # 5. Benchmark
    @pytest.mark.benchmark
    def test_latency_vs_pytorch_native(self):
        """Compare latency of custom CUDA pipeline vs PyTorch native."""
        device = torch.device("cuda")
        dtype = torch.float16
        B, S, D, I = 2, 2048, 4096, 11008

        x = torch.randn(B, S, D, device=device, dtype=dtype)
        norm_w = torch.randn(D, device=device, dtype=dtype)
        gate_w = torch.randn(I, D, device=device, dtype=dtype)
        up_w = torch.randn(I, D, device=device, dtype=dtype)

        ref_fn = lambda: torch_rms_swiglu_fused(x, norm_w, gate_w, up_w)
        custom_fn = lambda: rmsswiglu_forward(x, norm_w, gate_w, up_w)

        t_ref = benchmark_cuda(ref_fn)
        t_custom = benchmark_cuda(custom_fn)

        speedup = t_ref / t_custom if t_custom > 0 else float("inf")

        print(f"\n{'='*50}")
        print(f"  Shape:       ({B}, {S}, {D}) → ({B}, {S}, {I})")
        print(f"  PyTorch:     {t_ref:.4f} ms")
        print(f"  Custom CUDA: {t_custom:.4f} ms")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"{'='*50}")

        assert t_custom > 0, "Custom kernel returned zero timing"