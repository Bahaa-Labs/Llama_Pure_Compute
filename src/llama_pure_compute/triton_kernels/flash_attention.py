from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


# ============================================================================
# Tunable configurations
# ============================================================================

PREFILL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128},
        num_warps=8,
        num_stages=2,
    ),
]


DECODE_CONFIGS = [
    triton.Config(
        {"BLOCK_N": 64},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_N": 128},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_N": 128},
        num_warps=8,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_N": 256},
        num_warps=8,
        num_stages=2,
    ),
]


GENERAL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64},
        num_warps=4,
        num_stages=2,
    ),
]


# ============================================================================
# 1. Causal prefill
#
# Q_LEN == KV_LEN
#
# Optimizations:
# - square tiles
# - full blocks before diagonal are unmasked
# - only diagonal block receives causal masking
# - future blocks are never loaded
# ============================================================================

@triton.autotune(
    configs=PREFILL_CONFIGS,
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_causal_prefill_kernel(
    Q,
    K,
    V,
    O,
    sm_scale,

    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,

    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,

    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,

    stride_ob,
    stride_oh,
    stride_om,
    stride_od,

    H,

    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_id = pid_bh // H
    head_id = pid_bh % H

    q_base = (
        batch_id * stride_qb
        + head_id * stride_qh
    )

    k_base = (
        batch_id * stride_kb
        + head_id * stride_kh
    )

    v_base = (
        batch_id * stride_vb
        + head_id * stride_vh
    )

    o_base = (
        batch_id * stride_ob
        + head_id * stride_oh
    )

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_d = tl.arange(0, HEAD_DIM)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    q_ptrs = (
        Q
        + q_base
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )

    q_mask = (
        (offs_m[:, None] < SEQ_LEN)
        & (offs_d[None, :] < HEAD_DIM)
    )

    q = tl.load(
        q_ptrs,
        mask=q_mask,
        other=0.0,
    )

    # ------------------------------------------------------------------
    # Online softmax state
    # ------------------------------------------------------------------

    m_i = tl.full(
        (BLOCK_M,),
        float("-inf"),
        dtype=tl.float32,
    )

    l_i = tl.zeros(
        (BLOCK_M,),
        dtype=tl.float32,
    )

    acc = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )

    num_blocks: tl.constexpr = tl.cdiv(
        SEQ_LEN,
        BLOCK_N,
    )

    # ------------------------------------------------------------------
    # Fully visible blocks
    # ------------------------------------------------------------------

    for block_idx in range(0, num_blocks):
        if block_idx < pid_m:
            start_n = block_idx * BLOCK_N

            offs_n = (
                start_n
                + tl.arange(0, BLOCK_N)
            )

            # K: [D, N]
            k_ptrs = (
                K
                + k_base
                + offs_n[None, :] * stride_kn
                + offs_d[:, None] * stride_kd
            )

            k_mask = (
                (offs_d[:, None] < HEAD_DIM)
                & (offs_n[None, :] < SEQ_LEN)
            )

            k = tl.load(
                k_ptrs,
                mask=k_mask,
                other=0.0,
            )

            # V: [N, D]
            v_ptrs = (
                V
                + v_base
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )

            v_mask = (
                (offs_n[:, None] < SEQ_LEN)
                & (offs_d[None, :] < HEAD_DIM)
            )

            v = tl.load(
                v_ptrs,
                mask=v_mask,
                other=0.0,
            )

            # QK: [M, N]
            qk = tl.dot(
                q,
                k,
                out_dtype=tl.float32,
            )

            qk *= sm_scale

            m_ij = tl.maximum(
                m_i,
                tl.max(qk, axis=1),
            )

            p = tl.exp(
                qk - m_ij[:, None],
            )

            alpha = tl.exp(
                m_i - m_ij,
            )

            l_i = (
                l_i * alpha
                + tl.sum(p, axis=1)
            )

            acc = (
                acc * alpha[:, None]
                + tl.dot(
                    p.to(v.dtype),
                    v,
                    out_dtype=tl.float32,
                )
            )

            m_i = m_ij

    # ------------------------------------------------------------------
    # Diagonal block
    # ------------------------------------------------------------------

    start_n = pid_m * BLOCK_N

    offs_n = (
        start_n
        + tl.arange(0, BLOCK_N)
    )

    k_ptrs = (
        K
        + k_base
        + offs_n[None, :] * stride_kn
        + offs_d[:, None] * stride_kd
    )

    k_mask = (
        (offs_d[:, None] < HEAD_DIM)
        & (offs_n[None, :] < SEQ_LEN)
    )

    k = tl.load(
        k_ptrs,
        mask=k_mask,
        other=0.0,
    )

    v_ptrs = (
        V
        + v_base
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vd
    )

    v_mask = (
        (offs_n[:, None] < SEQ_LEN)
        & (offs_d[None, :] < HEAD_DIM)
    )

    v = tl.load(
        v_ptrs,
        mask=v_mask,
        other=0.0,
    )

    qk = tl.dot(
        q,
        k,
        out_dtype=tl.float32,
    )

    qk *= sm_scale

    causal_mask = (
        offs_m[:, None]
        >= offs_n[None, :]
    )

    valid_mask = (
        (offs_m[:, None] < SEQ_LEN)
        & (offs_n[None, :] < SEQ_LEN)
        & causal_mask
    )

    qk = tl.where(
        valid_mask,
        qk,
        float("-inf"),
    )

    m_ij = tl.maximum(
        m_i,
        tl.max(qk, axis=1),
    )

    p = tl.exp(
        qk - m_ij[:, None],
    )

    alpha = tl.exp(
        m_i - m_ij,
    )

    l_i = (
        l_i * alpha
        + tl.sum(p, axis=1)
    )

    acc = (
        acc * alpha[:, None]
        + tl.dot(
            p.to(v.dtype),
            v,
            out_dtype=tl.float32,
        )
    )

    m_i = m_ij

    # ------------------------------------------------------------------
    # Normalize + store
    # ------------------------------------------------------------------

    acc = acc / l_i[:, None]

    o_ptrs = (
        O
        + o_base
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )

    o_mask = (
        (offs_m[:, None] < SEQ_LEN)
        & (offs_d[None, :] < HEAD_DIM)
    )

    tl.store(
        o_ptrs,
        acc.to(q.dtype),
        mask=o_mask,
    )

# ============================================================================
# 2. Single-token decode
#
# Q_LEN == 1
#
# Shapes:
#   Q = [1, D]
#   K = [D, BLOCK_N]
#   V = [BLOCK_N, D]
#
# Therefore:
#   QK = [1, BLOCK_N]
#   P  = [1, BLOCK_N]
#   PV = [1, D]
# ============================================================================

@triton.autotune(
    configs=DECODE_CONFIGS,
    key=["KV_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_decode_kernel(
    Q,
    K,
    V,
    O,
    sm_scale,

    stride_qb,
    stride_qh,
    stride_qd,

    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,

    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,

    stride_ob,
    stride_oh,
    stride_od,

    H,

    KV_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,

    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    batch_id = pid_bh // H
    head_id = pid_bh % H

    q_base = (
        batch_id * stride_qb
        + head_id * stride_qh
    )

    k_base = (
        batch_id * stride_kb
        + head_id * stride_kh
    )

    v_base = (
        batch_id * stride_vb
        + head_id * stride_vh
    )

    o_base = (
        batch_id * stride_ob
        + head_id * stride_oh
    )

    offs_d = tl.arange(0, HEAD_DIM)

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------

    q_ptrs = (
        Q
        + q_base
        + offs_d * stride_qd
    )

    q = tl.load(
        q_ptrs,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    )

    # ------------------------------------------------------------------
    # Scalar online softmax state
    # ------------------------------------------------------------------

    m_i = tl.full(
        (1,),
        float("-inf"),
        dtype=tl.float32,
    )

    l_i = tl.zeros(
        (1,),
        dtype=tl.float32,
    )

    acc = tl.zeros(
        (1, HEAD_DIM),
        dtype=tl.float32,
    )

    num_blocks: tl.constexpr = tl.cdiv(
        KV_LEN,
        BLOCK_N,
    )

    # ------------------------------------------------------------------
    # KV-cache scan
    # ------------------------------------------------------------------

    for block_idx in range(0, num_blocks):

        start_n = (
            block_idx * BLOCK_N
        )

        offs_n = (
            start_n
            + tl.arange(0, BLOCK_N)
        )

        # --------------------------------------------------------------
        # K: [D, N]
        # --------------------------------------------------------------

        k_ptrs = (
            K
            + k_base
            + offs_n[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )

        k_mask = (
            (offs_d[:, None] < HEAD_DIM)
            & (offs_n[None, :] < KV_LEN)
        )

        k = tl.load(
            k_ptrs,
            mask=k_mask,
            other=0.0,
        )

        # --------------------------------------------------------------
        # V: [N, D]
        # --------------------------------------------------------------

        v_ptrs = (
            V
            + v_base
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )

        v_mask = (
            (offs_n[:, None] < KV_LEN)
            & (offs_d[None, :] < HEAD_DIM)
        )

        v = tl.load(
            v_ptrs,
            mask=v_mask,
            other=0.0,
        )

        # --------------------------------------------------------------
        # QK
        #
        # q_2d = [1, D]
        # k    = [D, N]
        # qk   = [1, N]
        # --------------------------------------------------------------

        q_2d = tl.reshape(
            q,
            (1, HEAD_DIM),
        )

        qk = tl.dot(
            q_2d,
            k,
            out_dtype=tl.float32,
        )

        qk *= sm_scale

        qk = tl.where(
            offs_n[None, :] < KV_LEN,
            qk,
            float("-inf"),
        )

        # --------------------------------------------------------------
        # Online softmax
        # --------------------------------------------------------------

        m_block = tl.max(
            qk,
            axis=1,
        )

        m_new = tl.maximum(
            m_i,
            m_block,
        )

        p = tl.exp(
            qk
            - m_new[:, None],
        )

        alpha = tl.exp(
            m_i
            - m_new,
        )

        l_i = (
            l_i * alpha
            + tl.sum(
                p,
                axis=1,
            )
        )

        # --------------------------------------------------------------
        # P @ V
        #
        # p = [1, N]
        # v = [N, D]
        # result = [1, D]
        # --------------------------------------------------------------

        acc_update = tl.dot(
            p.to(v.dtype),
            v,
            out_dtype=tl.float32,
        )

        acc = (
            acc * alpha[:, None]
            + acc_update
        )

        m_i = m_new

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    acc = acc / l_i[:, None]

    o_ptrs = (
        O
        + o_base
        + offs_d * stride_od
    )

    tl.store(
        o_ptrs,
        acc[0].to(q.dtype),
        mask=offs_d < HEAD_DIM,
    )


# ============================================================================
# 3. General fallback
# ============================================================================

@triton.autotune(
    configs=DECODE_CONFIGS,
    key=["KV_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_decode_kernel(
    Q,
    K,
    V,
    O,
    sm_scale,

    stride_qb,
    stride_qh,
    stride_qd,

    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,

    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,

    stride_ob,
    stride_oh,
    stride_od,

    H,

    KV_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,

    BLOCK_N: tl.constexpr,
):
    """
    Single-token autoregressive decode.

    Shapes:
        Q: [1, D]
        K: [D, BLOCK_N]
        V: [BLOCK_N, D]

        QK: [1, BLOCK_N]
        P:  [1, BLOCK_N]
        PV: [1, D]
        O:  [1, D]
    """

    pid_bh = tl.program_id(0)

    batch_id = pid_bh // H
    head_id = pid_bh % H

    q_base = (
        batch_id * stride_qb
        + head_id * stride_qh
    )

    k_base = (
        batch_id * stride_kb
        + head_id * stride_kh
    )

    v_base = (
        batch_id * stride_vb
        + head_id * stride_vh
    )

    o_base = (
        batch_id * stride_ob
        + head_id * stride_oh
    )

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # ------------------------------------------------------------------
    # Query: [D]
    # ------------------------------------------------------------------

    q_ptrs = (
        Q
        + q_base
        + offs_d * stride_qd
    )

    q = tl.load(
        q_ptrs,
        mask=offs_d < HEAD_DIM,
        other=0.0,
    )

    # Explicit 2-D query: [1, D]
    q_2d = tl.reshape(
        q,
        (1, HEAD_DIM),
    )

    # ------------------------------------------------------------------
    # Online softmax state
    # ------------------------------------------------------------------

    m_i = tl.full(
        (1,),
        float("-inf"),
        dtype=tl.float32,
    )

    l_i = tl.zeros(
        (1,),
        dtype=tl.float32,
    )

    acc = tl.zeros(
        (1, HEAD_DIM),
        dtype=tl.float32,
    )

    num_blocks: tl.constexpr = tl.cdiv(
        KV_LEN,
        BLOCK_N,
    )

    # ------------------------------------------------------------------
    # KV-cache scan
    # ------------------------------------------------------------------

    for block_idx in range(0, num_blocks):

        start_n = (
            block_idx * BLOCK_N
        )

        offs_n = (
            start_n
            + tl.arange(0, BLOCK_N)
        )

        # --------------------------------------------------------------
        # K: [D, N]
        # --------------------------------------------------------------

        k_ptrs = (
            K
            + k_base
            + offs_n[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )

        k_mask = (
            (offs_d[:, None] < HEAD_DIM)
            & (offs_n[None, :] < KV_LEN)
        )

        k = tl.load(
            k_ptrs,
            mask=k_mask,
            other=0.0,
        )

        # --------------------------------------------------------------
        # V: [N, D]
        # --------------------------------------------------------------

        v_ptrs = (
            V
            + v_base
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )

        v_mask = (
            (offs_n[:, None] < KV_LEN)
            & (offs_d[None, :] < HEAD_DIM)
        )

        v = tl.load(
            v_ptrs,
            mask=v_mask,
            other=0.0,
        )

        # --------------------------------------------------------------
        # QK: [1, N]
        # --------------------------------------------------------------

        qk = tl.dot(
            q_2d,
            k,
            out_dtype=tl.float32,
        )

        qk *= sm_scale

        valid_n = (
            offs_n[None, :] < KV_LEN
        )

        qk = tl.where(
            valid_n,
            qk,
            float("-inf"),
        )

        # --------------------------------------------------------------
        # Online softmax
        # --------------------------------------------------------------

        m_block = tl.max(
            qk,
            axis=1,
        )

        m_new = tl.maximum(
            m_i,
            m_block,
        )

        p = tl.exp(
            qk - m_new[:, None],
        )

        alpha = tl.exp(
            m_i - m_new,
        )

        l_i = (
            l_i * alpha
            + tl.sum(
                p,
                axis=1,
            )
        )

        # --------------------------------------------------------------
        # P @ V: [1, N] @ [N, D] -> [1, D]
        # --------------------------------------------------------------

        acc_update = tl.dot(
            p.to(v.dtype),
            v,
            out_dtype=tl.float32,
        )

        acc = (
            acc * alpha[:, None]
            + acc_update
        )

        m_i = m_new

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------

    acc = acc / l_i[:, None]

    o_ptrs = (
        O
        + o_base
        + offs_d[None, :] * stride_od
    )

    o_mask = (
        offs_d[None, :] < HEAD_DIM
    )

    tl.store(
        o_ptrs,
        acc.to(q.dtype),
        mask=o_mask,
    )


# ============================================================================
# Public API
# ============================================================================

def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    High-performance inference FlashAttention.

    Supported:
        Prefill:
            Q_LEN == KV_LEN

        Single-token decode:
            Q_LEN == 1 and KV_LEN > 1

        General:
            Other supported Q/K/V shapes
    """

    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError(
            "FlashAttention requires CUDA tensors."
        )

    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
    ):
        raise ValueError(
            "q, k, and v must be 4-D tensors."
        )

    if q.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise TypeError(
            "Only FP16 and BF16 are supported."
        )

    if (
        q.dtype != k.dtype
        or q.dtype != v.dtype
    ):
        raise TypeError(
            "q, k and v must have identical dtype."
        )

    if (
        q.device != k.device
        or q.device != v.device
    ):
        raise ValueError(
            "q, k and v must share the same device."
        )

    bq, hq, q_len, d = q.shape
    bk, hk, kv_len, dk = k.shape
    bv, hv, kv_len_v, dv = v.shape

    if bq != bk or bq != bv:
        raise ValueError(
            "Batch dimensions must match."
        )

    if hq != hk or hq != hv:
        raise ValueError(
            "Head counts must match."
        )

    if kv_len != kv_len_v:
        raise ValueError(
            "K and V sequence lengths must match."
        )

    if d != dk or d != dv:
        raise ValueError(
            "Head dimensions must match."
        )

    if q_len <= 0 or kv_len <= 0:
        raise ValueError(
            "Sequence lengths must be positive."
        )

    if q_len > kv_len:
        raise ValueError(
            "Q_LEN cannot exceed KV_LEN."
        )

    if d not in (
        32,
        64,
        128,
    ):
        raise ValueError(
            "Supported head dimensions: 32, 64, 128."
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if sm_scale is None:
        sm_scale = d**-0.5

    output = torch.empty_like(q)

    # ========================================================================
    # FAST PATH 1: causal prefill
    # ========================================================================

    if causal and q_len == kv_len:
        grid = lambda META: (
            triton.cdiv(
                q_len,
                META["BLOCK_M"],
            ),
            bq * hq,
        )

        _flash_causal_prefill_kernel[grid](
            q,
            k,
            v,
            output,
            sm_scale,

            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),

            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),

            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),

            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),

            hq,

            SEQ_LEN=q_len,
            HEAD_DIM=d,
        )

        return output

    # ========================================================================
    # FAST PATH 2: single-token decode
    # ========================================================================

    if causal and q_len == 1 and kv_len > 1:
        grid = (
            bq * hq,
        )

        _flash_decode_kernel[grid](
            q,
            k,
            v,
            output,
            sm_scale,

            q.stride(0),
            q.stride(1),
            q.stride(3),

            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),

            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),

            output.stride(0),
            output.stride(1),
            output.stride(3),

            hq,

            KV_LEN=kv_len,
            HEAD_DIM=d,
        )

        return output

    # ========================================================================
    # GENERAL FALLBACK
    # ========================================================================

    grid = lambda META: (
        triton.cdiv(
            q_len,
            META["BLOCK_M"],
        ),
        bq * hq,
    )

    _flash_general_kernel[grid](
        q,
        k,
        v,
        output,
        sm_scale,

        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),

        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),

        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),

        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),

        hq,

        Q_LEN=q_len,
        KV_LEN=kv_len,
        HEAD_DIM=d,
        IS_CAUSAL=causal,
    )

    return output


# ============================================================================
# Inference-only wrapper
# ============================================================================

class TritonFlashAttentionFunction(
    torch.autograd.Function
):
    """
    Forward-only wrapper.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        sm_scale: Optional[float] = None,
    ) -> torch.Tensor:

        return flash_attention_v2(
            q,
            k,
            v,
            causal=causal,
            sm_scale=sm_scale,
        )

    @staticmethod
    def backward(
        ctx,
        *grad_outputs,
    ):
        raise RuntimeError(
            "Backward is not implemented. "
            "Llama_Pure_Compute targets inference."
        )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:

    return TritonFlashAttentionFunction.apply(
        q,
        k,
        v,
        causal,
        sm_scale,
    )