import os 
import json
import math 
from typing import Optional, Tuple, Union, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_pure_compute.config import LlamaModelConfig
from llama_pure_compute.ops import rope_forward, rmsswiglu_forward
from llama_pure_compute.kv_manager import KVCacheManager
from llama_pure_compute.triton_kernels.rmsnorm import RMSNorm

# Conditional import for Triton FlashAttention kernel
_TRITON_FLASH_ATTN_AVAILABLE = False
try:
    from llama_pure_compute.triton_kernels.flash_attention import flash_attention_v2
    _TRITON_FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Precompute sin and cos frequency tables with RoPE (duplicated to match head_dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs), torch.sin(freqs)


class LlamaAttention(nn.Module):
    """Multi-Head & Grouped-Query Attention with Triton FlashAttention-v2 dispatch."""
    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.gqa_group_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCacheManager] = None,
        layer_idx: int = 0,
        mask: Optional[torch.Tensor] = None,
        use_triton_flash: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Linear Projections [B, S, H, D]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 2. Rotary Position Embeddings (RoPE)
        q, k = rope_forward(q, k, cos, sin, position_ids=positions)
        
        # 3. KV Cache Scatter Update
        if kv_cache is not None:
            start_pos = int(positions[0, 0].item()) if positions.ndim > 1 else int(positions[0].item())
            k, v = kv_cache.update(
                key_states=k,
                value_states=v,
                start_pos=start_pos,
                seq_len=seq_len
            )
            # Cache update outputs layout: [B, H_kv, S_cached, D]
        else:
            # Standard transpose to [B, H_kv, S_seq, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        q = q.transpose(1, 2)  # [B, H_q, S_seq, D]
        
        # 4. Grouped-Query Attention (GQA) Interleaving
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Determine causality requirement
        is_causal = (mask is None and seq_len > 1)

        # 5. Fast Path: Triton FlashAttention-v2 Dispatch
        can_use_triton = (
            _TRITON_FLASH_ATTN_AVAILABLE
            and use_triton_flash
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and mask is None 
        )

        if can_use_triton:
            output = flash_attention_v2(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v.contiguous(),
                causal=is_causal,
                sm_scale=self.scale,
            )
        # 6. Fallback Path: PyTorch SDPA or Manual Matmul
        elif hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=is_causal,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            output = torch.matmul(attn_weights, v)
        
        # 7. Reshape and Output Projection [B, S, H * D]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)

class LlamaDecodeLayer(nn.Module):
    # Transformer Block with Fused RMSNorm + SwiGLU by ops.py
    def __init__(self, config: LlamaModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        
        # SwiGLU Projections
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(config.dim, config.inter_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.inter_dim, bias=False)
        self.down_proj = nn.Linear(config.inter_dim, config.dim, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCacheManager] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1- Pre-LayerNorm Self-Attention
        h = x + self.self_attn(
            self.input_layernorm(x),
            positions=positions,
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            mask=mask,
        )
        
        # 2- Fused RMSNorm + SwiGLU MLP
        mlp_inter = rmsswiglu_forward(
            h,
            rms_weight=self.post_attention_layernorm.weight,
            gate_w=self.gate_proj.weight,
            up_w=self.up_proj.weight,
            eps=self.post_attention_layernorm.eps,
        )
        
        # 3- Down Projection + Residual
        out = h + self.down_proj(mlp_inter)
        return out


class LlamaModel(nn.Module):
    # Transformer with cached RoPE tables
    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [LlamaDecodeLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        
        # Precompute RoPE tables using config parameters
        cos, sin = precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len,
            theta=config.rope_theta
        )
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[KVCacheManager] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        
        # Gathering specific position cos/sin embeddings
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        
        if cos.ndim == 3:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        
        for layer in self.layers:
            h = layer(
                h,
                positions=positions,
                cos=cos,
                sin=sin,
                kv_cache=kv_cache,
                mask=mask,
            )
        return self.norm(h)


class LlamaForCausalLM(nn.Module):
    # Top-Level Model Class with checkpoint loading logic
    def __init__(self, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCacheManager] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if positions is None:
            positions = (
                torch.arange(0, seq_len, device=input_ids.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
        
        # Causal Mask for Prefill Phase
        if seq_len > 1 and mask is None:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_cache=kv_cache,
            mask=mask,
        )
        
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "LlamaForCausalLM":
        config_file = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file missing at {config_file}")
        
        config = LlamaModelConfig.from_json(config_file)
        model = cls(config).to(dtype=dtype, device="cpu")
        
        weight_files = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(".safetensors") or f.endswith(".bin")
        ]
        
        if not weight_files:
            raise FileNotFoundError(f"No .safetensors or .bin checkpoint files in {model_dir}")
        
        state_dict: Dict[str, torch.Tensor] = {}
        for wf in weight_files:
            if wf.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict.update(load_file(wf, device="cpu"))
            else:
                state_dict.update(torch.load(wf, map_location="cpu", weights_only=True))
        
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"missing weights: {result.missing_keys}")
        model = model.to(device=device, dtype=dtype)
        model.eval()
        
        return model