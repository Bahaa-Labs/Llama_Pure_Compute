from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_pure_compute.config import LlamaModelConfig
from llama_pure_compute.kv_manager import KVCacheManager
from llama_pure_compute.ops import rmsswiglu_forward, rope_forward
from llama_pure_compute.triton_kernels.rmsnorm import RMSNorm


try:
    from llama_pure_compute.triton_kernels.flash_attention import (
        flash_attention_v2,
    )

    _TRITON_FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attention_v2 = None
    _TRITON_FLASH_ATTN_AVAILABLE = False


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / dim
        )
    )

    positions = torch.arange(
        end,
        device=device,
        dtype=torch.float32,
    )

    freqs = torch.outer(
        positions,
        freqs,
    )

    freqs = torch.cat(
        [freqs, freqs],
        dim=-1,
    )

    return torch.cos(freqs), torch.sin(freqs)


class LlamaAttention(nn.Module):
    """
    Llama attention with GQA and prefill/decode dispatch.
    """

    def __init__(
        self,
        config: LlamaModelConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.gqa_group_size
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.dim,
            self.num_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            self.dim,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            self.dim,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCacheManager],
        layer_idx: int,
        mask: Optional[torch.Tensor] = None,
        use_triton_flash: bool = True,
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        k = self.k_proj(x).view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )

        v = self.v_proj(x).view(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )

        q, k = rope_forward(
            q=q,
            k=k,
            cos=cos,
            sin=sin,
            position_ids=positions,
        )

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if kv_cache is not None:
            start_pos = int(
                positions[0, 0].item()
            )

            k, v = kv_cache.update(
                layer_idx=layer_idx,
                key_states=k,
                value_states=v,
                start_pos=start_pos,
            )

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(
                self.num_kv_groups,
                dim=1,
            )

            v = v.repeat_interleave(
                self.num_kv_groups,
                dim=1,
            )

        kv_len = k.shape[-2]

        is_causal = mask is None

        can_use_triton = (
            use_triton_flash
            and _TRITON_FLASH_ATTN_AVAILABLE
            and q.is_cuda
            and q.dtype in (
                torch.float16,
                torch.bfloat16,
            )
            and mask is None
        )

        if can_use_triton:
            output = flash_attention_v2(
                q=q,
                k=k,
                v=v,
                causal=is_causal,
                sm_scale=self.scale,
            )

        elif hasattr(
            F,
            "scaled_dot_product_attention",
        ):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=is_causal,
                scale=self.scale,
            )

        else:
            scores = torch.matmul(
                q,
                k.transpose(-2, -1),
            ) * self.scale

            if mask is not None:
                scores = scores + mask

            probs = F.softmax(
                scores,
                dim=-1,
                dtype=torch.float32,
            ).to(q.dtype)

            output = torch.matmul(
                probs,
                v,
            )

        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                -1,
            )
        )

        return self.o_proj(output)


class LlamaDecodeLayer(nn.Module):
    def __init__(
        self,
        config: LlamaModelConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(
            config.dim,
            eps=config.rms_norm_eps,
        )

        self.self_attn = LlamaAttention(
            config,
        )

        self.post_attention_layernorm = RMSNorm(
            config.dim,
            eps=config.rms_norm_eps,
        )

        self.gate_proj = nn.Linear(
            config.dim,
            config.inter_dim,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.dim,
            config.inter_dim,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.inter_dim,
            config.dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCacheManager],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:

        h = x + self.self_attn(
            self.input_layernorm(x),
            positions=positions,
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            mask=mask,
        )

        mlp_inter = rmsswiglu_forward(
            h,
            rms_weight=self.post_attention_layernorm.weight,
            gate_w=self.gate_proj.weight,
            up_w=self.up_proj.weight,
            eps=self.post_attention_layernorm.eps,
        )

        return h + self.down_proj(
            mlp_inter
        )


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaModelConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.dim,
        )

        self.layers = nn.ModuleList(
            [
                LlamaDecodeLayer(
                    config,
                    layer_idx=i,
                )
                for i in range(config.num_layers)
            ]
        )

        self.norm = RMSNorm(
            config.dim,
            eps=config.rms_norm_eps,
        )

        cos, sin = precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len,
            theta=config.rope_theta,
        )

        self.register_buffer(
            "cos_cached",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            sin,
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor,
        kv_cache: Optional[KVCacheManager],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:

        h = self.embed_tokens(
            input_ids
        )

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
    def __init__(
        self,
        config: LlamaModelConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.model = LlamaModel(
            config
        )

        self.lm_head = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        positions: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCacheManager] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size, seq_len = input_ids.shape

        if positions is None:
            positions = (
                torch.arange(
                    seq_len,
                    device=input_ids.device,
                )
                .unsqueeze(0)
                .expand(
                    batch_size,
                    -1,
                )
            )

        if mask is None and kv_cache is None:
            if seq_len > 1:
                mask = torch.full(
                    (
                        seq_len,
                        seq_len,
                    ),
                    float("-inf"),
                    device=input_ids.device,
                    dtype=torch.float32,
                )

                mask = torch.triu(
                    mask,
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)

        hidden_states = self.model(
            input_ids,
            positions=positions,
            kv_cache=kv_cache,
            mask=mask,
        )

        return self.lm_head(
            hidden_states
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "LlamaForCausalLM":

        config_file = os.path.join(
            model_dir,
            "config.json",
        )

        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"Missing config: {config_file}"
            )

        config = LlamaModelConfig.from_json(
            config_file
        )

        model = cls(
            config
        )

        model = model.to(
            device="cpu",
            dtype=dtype,
        )

        weight_files = [
            os.path.join(
                model_dir,
                filename,
            )
            for filename in os.listdir(model_dir)
            if filename.endswith(
                (
                    ".safetensors",
                    ".bin",
                )
            )
        ]

        if not weight_files:
            raise FileNotFoundError(
                f"No checkpoint weights found in {model_dir}"
            )

        state_dict: Dict[str, torch.Tensor] = {}

        for weight_file in weight_files:
            if weight_file.endswith(
                ".safetensors"
            ):
                from safetensors.torch import load_file

                state_dict.update(
                    load_file(
                        weight_file,
                        device="cpu",
                    )
                )
            else:
                state_dict.update(
                    torch.load(
                        weight_file,
                        map_location="cpu",
                        weights_only=True,
                    )
                )

        result = model.load_state_dict(
            state_dict,
            strict=False,
        )

        if result.missing_keys:
            raise RuntimeError(
                "Missing checkpoint weights:\n"
                + "\n".join(
                    result.missing_keys
                )
            )

        if result.unexpected_keys:
            raise RuntimeError(
                "Unexpected checkpoint weights:\n"
                + "\n".join(
                    result.unexpected_keys
                )
            )

        model = model.to(
            device=device,
            dtype=dtype,
        )

        model.eval()

        return model