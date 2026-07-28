import torch
import torch.nn as nn
from typing import Optional, TYPE_CHECKING

# 1. Guarded Import for Environments without Triton/CUDA
if TYPE_CHECKING:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
else:
    try:
        import triton
        import triton.language as tl
        TRITON_AVAILABLE = True
    except ImportError:
        triton = None
        tl = None

if TRITON_AVAILABLE:
    @triton.jit
    def _rmsnorm_triton_kernel(
        X_ptr,          
        Y_ptr,          
        W_ptr,          
        stride_x_row,  
        stride_y_row,  
        N_cols,         
        eps,            
        BLOCK_SIZE: "tl.constexpr",
    ):
        row_idx = tl.program_id(0)
        
        # Calculating row memory offsets
        X_row_ptr = X_ptr + row_idx * stride_x_row
        Y_row_ptr = Y_ptr + row_idx * stride_y_row

        # Offsets across hidden dimensions
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N_cols

        # Load input values into Registers (cast to float32 for variance stability)
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Compute Mean Square
        mean_sq = tl.sum(x * x, axis=0) / N_cols
        rsqrt = tl.math.rsqrt(mean_sq + eps)
        
        # Normalize and scale by weight
        y = x * rsqrt * w

        y_out = y.to(X_ptr.dtype.element_ty)

        # Store output with spatial masking
        tl.store(Y_row_ptr + cols, y_out, mask=mask)


def rmsnorm_triton_forward(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    # Ensure input tensor is contiguous for raw pointer arithmetic
    x = x.contiguous()
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    N_rows, N_cols = x_2d.shape

    out_2d = torch.empty_like(x_2d)
    
    # Calculate SRAM block size
    BLOCK_SIZE = triton.next_power_of_2(N_cols)
    
    # 1D Grid where each block handles 1 row
    grid = (N_rows,)
    _rmsnorm_triton_kernel[grid](
        x_2d,
        out_2d,
        weight,
        x_2d.stride(0),
        out_2d.stride(0),
        N_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_2d.view(*orig_shape)


@torch.jit.script
def _rmsnorm_pytorch_fallback(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(variance + eps)
    return ((x_fp32 * rsqrt) * weight.to(torch.float32)).to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected hidden dimension {self.dim}, got {x.shape[-1]}"
            )
        
        # Dispatch to Triton if available and running on CUDA
        if TRITON_AVAILABLE and x.is_cuda:
            return rmsnorm_triton_forward(x, self.weight, self.eps)
        
        return _rmsnorm_pytorch_fallback(x, self.weight, self.eps)
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# Verification Harness
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size, seq_len, hidden_dim = 2, 8, 4096
    norm = RMSNorm(hidden_dim).to(device=device, dtype=torch.float16)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16)
    
    out = norm(x)
    print(f"Device: {device} | Triton Active: {TRITON_AVAILABLE and x.is_cuda}")
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    
    assert out.shape == x.shape, "Shape mismatch!"
    assert not torch.isnan(out).any(), "NaN values detected!"
    print("Triton RMSNorm test execution successful!")