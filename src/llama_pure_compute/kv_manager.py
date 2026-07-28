from __future__ import annotations
import logging 
from typing import Optional, Tuple, Union
import torch

from llama_pure_compute.ops import rope_forward, update_kv_cache

logger = logging.getLogger("llama_pure_compute.kv_manager")

class KVCacheManager:
    """Manage static pre-allocated KV Caches for continuous decoding using custom CUDA kernels."""
    def __init__(
        self, 
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("KVCacheManager initialized on CUDA but CUDA is not available.")
        
        # Native PyTorch 4D Attention Layout [B, H, S, D]
        self.k_cache: torch.Tensor = torch.zeros(
            (max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=self.dtype, device=self.device,
        )
        self.v_cache: torch.Tensor = torch.zeros(
            (max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=self.dtype, device=self.device,
        )
        
        logger.info(
            f"Initialized KVCacheManager [Batch: {max_batch_size}, SeqLen: {max_seq_len}, "
            f"Heads: {n_kv_heads}, Dim: {head_dim}] | Memory: {self._get_memory_footprint_mb():.2f} MB"
        )
        
    def _get_memory_footprint_mb(self) -> float:
        bytes_per_element = torch.tensor([], dtype=self.dtype).element_size()
        total_elements = self.k_cache.numel() + self.v_cache.numel()
        return (total_elements * bytes_per_element) / (1024 * 1024)

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,   
        value_states: torch.Tensor, 
        start_pos: int,
        seq_len: int,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scatter-updates new K and V states into static 4D cache buffers.
        """
        bsz = key_states.shape[0]
        
        if bsz > self.max_batch_size:
            raise ValueError(f"Batch size {bsz} exceeds maximum pre-allocated batch size {self.max_batch_size}")
        if slot_mapping is None and (start_pos + seq_len > self.max_seq_len):
            raise ValueError(f"Sequence position {start_pos + seq_len} exceeds max sequence length {self.max_seq_len}")
        
        # Source input states: [bsz, n_kv_heads, seq_len, head_dim] -> [num_tokens, n_kv_heads, head_dim]
        k_src = key_states.permute(0, 2, 1, 3).reshape(-1, self.n_kv_heads, self.head_dim).contiguous()
        v_src = value_states.permute(0, 2, 1, 3).reshape(-1, self.n_kv_heads, self.head_dim).contiguous()

        if slot_mapping is None:
            batch_offsets = (
                torch.arange(bsz, device=key_states.device, dtype=torch.int64) * self.max_seq_len
            ).unsqueeze(1)
            pos_offsets = torch.arange(
                start_pos, start_pos + seq_len, device=key_states.device, dtype=torch.int64
            ).unsqueeze(0)
            
            slot_mapping_tensor = (batch_offsets + pos_offsets).view(-1)
        else:
            slot_mapping_tensor = slot_mapping

        if key_states.is_cuda and self.k_cache.is_cuda:
            update_kv_cache(
                key_src=k_src,
                value_src=v_src,
                key_cache=self.k_cache,
                value_cache=self.v_cache,
                slot_mapping=slot_mapping_tensor,
            )
        else:
            # CPU Fallback: Convert 1D slot indices to explicit 4D [b, h, s, d] indexing
            num_tokens = k_src.shape[0]
            batch_indices = slot_mapping_tensor // self.max_seq_len
            seq_indices = slot_mapping_tensor % self.max_seq_len

            for i in range(num_tokens):
                b = batch_indices[i].item()
                s = seq_indices[i].item()
                if b < 0 or s < 0:
                    continue
                self.k_cache[b, :, s, :] = k_src[i]
                self.v_cache[b, :, s, :] = v_src[i]

        keys_out = self.k_cache[:bsz, :, : start_pos + seq_len, :]
        values_out = self.v_cache[:bsz, :, : start_pos + seq_len, :]
        
        return keys_out, values_out

    @torch.no_grad()
    def update_with_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,   
        value_states: torch.Tensor, 
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_rot, k_rot = rope_forward(
            q=query_states,
            k=key_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
        )
        
        k_cached, v_cached = self.update(
            key_states=k_rot,
            value_states=value_states,
            start_pos=start_pos,
            seq_len=seq_len,
            slot_mapping=slot_mapping,
        )
        
        return q_rot, k_cached, v_cached

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        logger.debug("KV Cache memory pool reset.")