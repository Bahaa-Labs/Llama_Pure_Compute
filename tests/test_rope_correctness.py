import pytest 
import torch
from llama_pure_compute.ops import rope_forward

# Pytorch Reference Implementation
def reference_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q shape: [batch_size, num_tokens, num_heads, head_dim]
    k shape: [batch_size, num_tokens, num_kv_heads, head_dim]
    cos/sin shape: [max_seq_len, half_rotary]
    """
    q_out = q.clone().float()
    k_out = k.clone().float()
    head_dim = q.shape[-1]
    half_rotary = head_dim // 2
    
    # Reshape input to isolate lower/upper half-rotary split
    # q0: [..., :half_rotary], q1: [..., half_rotary:]
    q0, q1 = q_out[..., :half_rotary], q_out[..., half_rotary:]
    k0, k1 = k_out[..., :half_rotary], k_out[..., half_rotary:]
    
    # Resolve Position indices
    num_tokens = q.numel() // (q.shape[-2] * head_dim)
    if position_ids is not None:
        pos_flat = position_ids.view(-1)
    else:
        pos_flat = torch.arange(num_tokens, device=q.device, dtype=torch.long)
    
    # Retrieve Frequencies: [num_tokens, 1, half_rotary] for broadcasting heads
    c = cos[pos_flat].view(num_tokens, 1, half_rotary)
    s = sin[pos_flat].view(num_tokens, 1, half_rotary)
    
    # Reshape Q/K to align flattened token indexing
    q0_flat = q0.view(num_tokens, -1, half_rotary)
    q1_flat = q1.view(num_tokens, -1, half_rotary)
    k0_flat = k0.view(num_tokens, -1, half_rotary)
    k1_flat = k1.view(num_tokens, -1, half_rotary)
    
    # Apply 2D rotation matrix [[cos, -sin], [sin, cos]]
    q_rot_0 = q0_flat * c - q1_flat * s
    q_rot_1 = q0_flat * s + q1_flat * c
    k_rot_0 = k0_flat * c - k1_flat * s
    k_rot_1 = k0_flat * s + k1_flat * c

    # Construct final rotated tensor
    q_out_final = torch.cat([q_rot_0, q_rot_1], dim=-1).view_as(q).to(q.dtype)
    k_out_final = torch.cat([k_rot_0, k_rot_1], dim=-1).view_as(k).to(k.dtype)
    
    return q_out_final, k_out_final

# Test Helpers & Fixtures
def generate_rope_freqs(max_seq_len: int, head_dim: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    half_rotary = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_rotary, dtype=torch.float32, device=device) / half_rotary))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

# Test Suite
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "num_heads, num_kv_heads",
    [
        (32, 32), # MQA
        (32, 8), # GQA
        (64, 8), # Extreme GQA (8: 1)
    ]
)
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 1), # single token decoding step
        (1, 128), # single batch prefill
        (4, 512), # multi batch long context
    ]
)
def test_rope_correctness(
    dtype: torch.dtype,
    num_heads: int,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
):
    device = "cuda"
    head_dim = 128
    max_seq_len = 2048
    
    # Generating test inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    cos, sin = generate_rope_freqs(max_seq_len, head_dim, device=device)
    
    # Cast frequencies to tensor dtype for kernel consumption
    cos_dt = cos.to(dtype)
    sin_dt = sin.to(dtype)
    
    # 1- Compute Pytorch Reference
    q_ref, k_ref = reference_rope(q, k, cos_dt, sin_dt)
    
    # 2- Compute CUDA Kernel output
    q_cuda = q.clone()
    k_cuda = k.clone()
    q_out_cuda, k_out_cuda = rope_forward(q_cuda, k_cuda, cos_dt, sin_dt, None)
    
    # 3- Precision Tolerance Thresholds
    tolerances = {
        torch.float32: {"rtol": 1e-5, "atol": 1e-5},
        torch.float16: {"rtol": 1e-3, "atol": 1e-3},
        torch.bfloat16: {"rtol": 1.5e-2, "atol": 1.5e-2},
    }   
    tol = tolerances[dtype]
    
    # 4- Assertions
    torch.testing.assert_close(q_out_cuda, q_ref, rtol=tol["rtol"], atol=tol["atol"])
    torch.testing.assert_close(k_out_cuda, k_ref, rtol=tol["rtol"], atol=tol["atol"])
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_rope_custom_position_ids():
    # Verify handling non-contiguous sparse position_ids
    device = "cuda"
    dtype = torch.float16
    batch_size, seq_len, num_heads, num_kv_heads, head_dim = 2, 16, 8, 2, 128    
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    cos, sin = generate_rope_freqs(1024, head_dim, device=device)
    
    position_ids = torch.randint(10, 500, (batch_size, seq_len), device=device, dtype=torch.long)
    
    q_ref, k_ref = reference_rope(q, k, cos.to(dtype), sin.to(dtype), position_ids=position_ids)
    
    q_cuda = q.clone()
    k_cuda = k.clone()
    q_out_cuda, k_out_cuda = rope_forward(q_cuda, k_cuda, cos.to(dtype), sin.to(dtype), position_ids)
    
    torch.testing.assert_close(q_out_cuda, q_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_out_cuda, k_ref, rtol=1e-3, atol=1e-3)
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_rope_in_place_mutation():
    device = "cuda"
    q = torch.randn(1, 16, 8, 128, device=device, dtype=torch.float32)
    k = torch.randn(1, 16, 2, 128, device=device, dtype=torch.float32)
    cos, sin = generate_rope_freqs(256, 128, device=device)
    
    q_ptr_before = q.data_ptr()
    k_ptr_before = k.data_ptr()
    q_out, k_out = rope_forward(q, k, cos, sin, None)
    
    # Zero Memory Allocation overhead
    assert q_out.data_ptr() == q_ptr_before
    assert k_out.data_ptr() == k_ptr_before
    
    
    