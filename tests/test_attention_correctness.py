from __future__ import annotations

import torch
import pytest
import torch.nn.functional as F

from llama_pure_compute.triton_kernels.flash_attention import flash_attention_v2


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """
    FP32 reference implementation.

    Supports both:
      Q_LEN == KV_LEN  -> prefill
      Q_LEN != KV_LEN  -> decode / cached attention
    """
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    q_len = q.shape[-2]
    kv_len = k.shape[-2]

    scores = torch.matmul(
        q_f32,
        k_f32.transpose(-2, -1),
    ) * sm_scale

    if causal:
        # Absolute query positions are the final q_len positions
        # in the KV sequence.
        q_positions = (
            torch.arange(
                kv_len - q_len,
                kv_len,
                device=q.device,
            )
            .view(1, 1, q_len, 1)
        )

        kv_positions = (
            torch.arange(
                kv_len,
                device=q.device,
            )
            .view(1, 1, 1, kv_len)
        )

        causal_mask = kv_positions > q_positions

        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

    probs = F.softmax(
        scores,
        dim=-1,
    )

    output = torch.matmul(
        probs,
        v_f32,
    )

    return output.to(q.dtype)


def assert_attention_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    dtype: torch.dtype,
    description: str,
) -> None:
    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2

    max_abs_err = (
        torch.max(
            torch.abs(
                actual.float() - expected.float()
            )
        )
        .item()
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=True,
        check_dtype=True,
        msg=(
            f"{description} | "
            f"Max Abs Err: {max_abs_err:.6f}"
        ),
    )


# ---------------------------------------------------------------------------
# Prefill correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [128, 512, 2048])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
def test_flash_attention_prefill_correctness(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda"

    scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        dtype=dtype,
        device=device,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = reference_attention(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
    )

    actual = flash_attention_v2(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
    )

    assert_attention_close(
        actual,
        expected,
        dtype=dtype,
        description=(
            f"Prefill B={batch_size}, "
            f"H={num_heads}, "
            f"S={seq_len}, "
            f"D={head_dim}, "
            f"causal={causal}, "
            f"dtype={dtype}"
        ),
    )


# ---------------------------------------------------------------------------
# Unaligned prefill
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)
@pytest.mark.parametrize(
    "seq_len",
    [127, 357, 513, 1025],
)
def test_flash_attention_unaligned_prefill(
    seq_len: int,
) -> None:
    torch.manual_seed(1337)

    q = torch.randn(
        2,
        4,
        seq_len,
        64,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = reference_attention(
        q,
        k,
        v,
        causal=False,
    )

    actual = flash_attention_v2(
        q,
        k,
        v,
        causal=False,
    )

    assert_attention_close(
        actual,
        expected,
        dtype=torch.float16,
        description=f"Unaligned prefill S={seq_len}",
    )


# ---------------------------------------------------------------------------
# Decode / KV-cache correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize(
    "kv_len",
    [128, 512, 2048],
)
@pytest.mark.parametrize(
    "head_dim",
    [64, 128],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
def test_flash_attention_decode_correctness(
    batch_size: int,
    num_heads: int,
    kv_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    device = "cuda"

    scale = 1.0 / (head_dim ** 0.5)

    # One query token attending over an existing KV cache.
    q = torch.randn(
        batch_size,
        num_heads,
        1,
        head_dim,
        dtype=dtype,
        device=device,
    )

    k = torch.randn(
        batch_size,
        num_heads,
        kv_len,
        head_dim,
        dtype=dtype,
        device=device,
    )

    v = torch.randn_like(k)

    expected = reference_attention(
        q,
        k,
        v,
        causal=True,
        sm_scale=scale,
    )

    actual = flash_attention_v2(
        q,
        k,
        v,
        causal=True,
        sm_scale=scale,
    )

    assert_attention_close(
        actual,
        expected,
        dtype=dtype,
        description=(
            f"Decode B={batch_size}, "
            f"H={num_heads}, "
            f"Q=1, "
            f"KV={kv_len}, "
            f"D={head_dim}, "
            f"dtype={dtype}"
        ),
    )


# ---------------------------------------------------------------------------
# Multi-token decode correctness
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)
@pytest.mark.parametrize(
    "q_len,kv_len",
    [
        (1, 128),
        (1, 512),
        (1, 2048),
        (2, 512),
        (4, 512),
        (4, 2048),
        (8, 2048),
    ],
)
def test_flash_attention_q_len_less_than_kv_len(
    q_len: int,
    kv_len: int,
) -> None:
    torch.manual_seed(2026)

    q = torch.randn(
        1,
        8,
        q_len,
        64,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(
        1,
        8,
        kv_len,
        64,
        dtype=torch.float16,
        device="cuda",
    )

    v = torch.randn_like(k)

    expected = reference_attention(
        q,
        k,
        v,
        causal=True,
    )

    actual = flash_attention_v2(
        q,
        k,
        v,
        causal=True,
    )

    assert_attention_close(
        actual,
        expected,
        dtype=torch.float16,
        description=(
            f"Q<KV Q={q_len}, KV={kv_len}"
        ),
    )