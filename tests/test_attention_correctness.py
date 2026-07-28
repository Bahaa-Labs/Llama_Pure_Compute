import torch 
import pytest
import torch.nn.functional as F
from llama_pure_compute.triton_kernels.flash_attention import flash_attention_v2

def reference_attension(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None
) -> torch.Tensor:
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        
    # Convert to FP32 for reference high precision numerical ground truth
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)
    
    # Compute Attention Scores: [B, H, N_CTX, N_CTX]
    scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * sm_scale
    if causal:
        seq_len = q.shape[2]
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=q.device),
            diagonal=1
        )
        scores = scores + mask
    
    attn_probs = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_probs, v_f32)
    return output.to(q.dtype)

# Parameterizing Pytest Harness
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("seq_len", [128, 512, 2048])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attension_v2_correctness(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    sm_scale = 1.0 / (head_dim ** 0.5)
    device = "cuda"
    
    q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    
    torch.cuda.synchronize()
    
    ref_out = reference_attension(q, k, v, causal=causal, sm_scale=sm_scale)
    triton_out = flash_attention_v2(q, k, v, causal=causal, sm_scale=sm_scale)
    
    torch.cuda.synchronize()
    
    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    
    max_abs_err = torch.max(torch.abs(ref_out - triton_out)).item()
    
    torch.testing.assert_close(
        triton_out,
        ref_out,
        rtol=rtol,
        atol=atol,
        check_device=True,
        check_dtype=True,
        msg=f"Divergence in Config: B={batch_size}, H={num_heads}, N={seq_len}, D={head_dim}, Causal={causal}, Dtype={dtype} | Max Abs Err: {max_abs_err:.6f}"
    )

# Boundary Verification
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_flash_attension_unaligned_seq_len():
    torch.manual_seed(1337)
    
    q = torch.randn((2, 4, 357, 64), dtype=torch.float16, device="cuda")
    k = torch.randn((2, 4, 357, 64), dtype=torch.float16, device="cuda")
    v = torch.randn((2, 4, 357, 64), dtype=torch.float16, device="cuda")
    
    ref_out = reference_attension(q, k, v, causal=False)
    triton_out = flash_attention_v2(q, k, v, causal=False)
    
    torch.cuda.synchronize()
    torch.testing.assert_close(triton_out, ref_out, rtol=1e-2, atol=1e-2)