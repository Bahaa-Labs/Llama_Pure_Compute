from __future__ import annotations
import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F

# Setup Logger for CUDA Extension Fallback Alerts
logger = logging.getLogger("llama_pure_compute.ops")

# Flag to track whether custom CUDA kernels are loaded and available
_CUDA_KERNELS_AVAILABLE = False 

try:
    from llama_pure_compute import _C as _backend
    _CUDA_KERNELS_AVAILABLE = True
except ImportError:
    # Fallback warning if extension isn't compiled
    import warnings
    warnings.warn(
        "Llama_Pure_Compute CUDA backend (_C) is not installed/loaded. "
        "Falling back to standard PyTorch implementation.",
        RuntimeWarning
    )


# PyTorch Native Fallbacks for CPU testing and non-CUDA envs
def _rope_forward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # PyTorch fallback implementation for Rotary Position Embeddings.
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _rmsswiglu_forward_pytorch(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    # PyTorch fallback for fused RMSNorm + SwiGLU MLP block.
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps) * rms_weight

    gate = F.linear(x_norm, gate_w)
    up = F.linear(x_norm, up_w)

    return F.silu(gate) * up


def _update_kv_cache_pytorch(
    key_src: torch.Tensor,
    value_src: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: Optional[torch.Tensor] = None,
) -> None:
    """
    PyTorch native fallback for scatter-updating the KV cache.
    """
    num_tokens = key_src.shape[0] if key_src.ndim == 3 else key_src.numel() // (key_src.shape[-2] * key_src.shape[-1])
    
    # Reshape key_src and value_src to flat token layout [num_tokens, num_heads, head_dim]
    k_flat = key_src.view(-1, key_src.shape[-2], key_src.shape[-1])
    v_flat = value_src.view(-1, value_src.shape[-2], value_src.shape[-1])
    
    # Reshape cache buffers to flat slot representation if needed [total_slots, num_heads, head_dim]
    cache_k_flat = key_cache.view(-1, key_cache.shape[-2], key_cache.shape[-1])
    cache_v_flat = value_cache.view(-1, value_cache.shape[-2], value_cache.shape[-1])

    if slot_mapping is None:
        slot_mapping = torch.arange(num_tokens, device=key_src.device, dtype=torch.int64)

    # Perform scatter update across valid slot IDs
    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    
    if valid_slots.numel() > 0:
        cache_k_flat[valid_slots] = k_flat[valid_mask]
        cache_v_flat[valid_slots] = v_flat[valid_mask]


# Public API
def rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if _CUDA_KERNELS_AVAILABLE and q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda:
        q_contig = q.contiguous()
        k_contig = k.contiguous()
        cos_contig = cos.contiguous()
        sin_contig = sin.contiguous()
        pos_contig = position_ids.contiguous() if position_ids is not None else None

        return _backend.rope_forward(
            q_contig, 
            k_contig, 
            cos_contig, 
            sin_contig, 
            pos_contig
        )

    return _rope_forward_pytorch(q, k, cos, sin, position_ids)


def rmsswiglu_forward(
    x: torch.Tensor,
    rms_weight: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    if _CUDA_KERNELS_AVAILABLE and x.is_cuda and rms_weight.is_cuda:
        return _backend.rmsswiglu_forward(
            x.contiguous(),
            rms_weight.contiguous(),
            gate_w.contiguous(),
            up_w.contiguous(),
            eps
        )

    return _rmsswiglu_forward_pytorch(x, rms_weight, gate_w, up_w, eps)


def update_kv_cache(
    key_src: torch.Tensor,
    value_src: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: Optional[torch.Tensor] = None,
) -> None:
    """
    Scatter-updates new Key/Value tokens into the flat/paged KV Cache.
    Dispatches to custom CUDA kernel if available; otherwise uses PyTorch fallback.
    """
    if _CUDA_KERNELS_AVAILABLE and key_src.is_cuda and key_cache.is_cuda:
        k_src_contig = key_src.contiguous()
        v_src_contig = value_src.contiguous()
        k_cache_contig = key_cache.contiguous()
        v_cache_contig = value_cache.contiguous()
        slot_contig = slot_mapping.contiguous() if slot_mapping is not None else None

        _backend.update_kv_cache(
            k_src_contig,
            v_src_contig,
            k_cache_contig,
            v_cache_contig,
            slot_contig
        )
        return

    _update_kv_cache_pytorch(key_src, value_src, key_cache, value_cache, slot_mapping)


def is_cuda_backend_available() -> bool:
    return _CUDA_KERNELS_AVAILABLE