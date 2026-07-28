"""
Features:
- Dual-loop execution (unmasked full blocks + single causal diagonal block).
- Memory pointer striding optimization.
- Autotuning targeted at GA102 SRAM/Register ratios.
"""
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32}, num_warps=2, num_stages=2),
    ],
    key=['N_CTX', 'HEAD_DIM', 'IS_CAUSAL'],
)
@triton.jit
def _flash_attn_v2_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_in_group = 8  # Group size for L2 cache swizzling
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_in_group
    group_size_m = min(num_pid_m - first_pid_m, num_pid_in_group)
    start_m = first_pid_m + (pid % group_size_m)
    
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Setup base tensor pointers for batch-head slice
    q_offset = off_z * stride_qb + off_h * stride_qh
    k_offset = off_z * stride_kb + off_h * stride_kh
    v_offset = off_z * stride_vb + off_h * stride_vh
    o_offset = off_z * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block into registers and scale up-front
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    q = (q * sm_scale).to(tl.float16)

    # Statistical accumulators (FP32 precision)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Initial K and V pointers
    offs_n_init = tl.arange(0, BLOCK_N)
    k_ptrs = K + k_offset + (offs_n_init[None, :] * stride_kn + offs_d[:, None] * stride_kd)
    v_ptrs = V + v_offset + (offs_n_init[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # Determine loop bounds
    if IS_CAUSAL:
        tc = start_m * BLOCK_M
        full_blocks = tc // BLOCK_N
        has_diagonal = True
    else:
        full_blocks = tl.cdiv(N_CTX, BLOCK_N)
        has_diagonal = False

    # 1. UNMASKED LOOP (Full blocks strictly below causal boundary)
    for block_idx in range(0, full_blocks):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        # Online Softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_ij

        # Advance memory pointers using strides
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # 2. DIAGONAL BLOCK (Causal Masking Applied Only Here)
    if has_diagonal:
        start_n = full_blocks * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(k_ptrs, mask=offs_n[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        # Mask elements above the causal diagonal
        mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_ij

    # Epilogue: Normalize and write back
    acc = acc / l_i[:, None]

    o_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None
) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors."

    Z, H, N_CTX, HEAD_DIM = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    out = torch.empty_like(q)

    grid = lambda META: (
        triton.cdiv(N_CTX, META['BLOCK_M']),
        Z * H
    )

    _flash_attn_v2_fwd_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        IS_CAUSAL=causal,
    )

    return out


class TritonFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True, sm_scale=None):
        return flash_attention_v2(q, k, v, causal=causal, sm_scale=sm_scale)


def flash_attn_func(q, k, v, causal=True, sm_scale=None):
    return TritonFlashAttentionFunction.apply(q, k, v, causal, sm_scale)
