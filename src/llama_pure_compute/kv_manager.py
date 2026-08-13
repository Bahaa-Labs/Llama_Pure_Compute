from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from llama_pure_compute.ops import update_kv_cache

logger = logging.getLogger("llama_pure_compute.kv_manager")


@dataclass(frozen=True)
class KVCacheSpec:
    num_layers: int
    max_batch_size: int
    max_seq_len: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device


class KVCacheManager:
    """
    Per-layer static KV-cache manager.

    The cache is allocated once and reused across requests.

    Layout per layer:
        [batch, kv_heads, sequence, head_dim]

    This implementation supports:
        - batched prefill with a shared starting position
        - autoregressive decode
        - request reset without clearing the whole GPU allocation
        - explicit per-layer cache ownership

    It intentionally does not implement continuous batching yet.
    That belongs to the next runtime stage.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if n_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("n_kv_heads and head_dim must be positive")

        self.spec = KVCacheSpec(
            num_layers=num_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=torch.device(device),
        )

        if (
            self.spec.device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "KVCacheManager requested CUDA but CUDA is unavailable."
            )

        self.device = self.spec.device

        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []

        for _ in range(num_layers):
            self.k_cache.append(
                torch.empty(
                    (
                        max_batch_size,
                        n_kv_heads,
                        max_seq_len,
                        head_dim,
                    ),
                    dtype=dtype,
                    device=self.device,
                )
            )

            self.v_cache.append(
                torch.empty(
                    (
                        max_batch_size,
                        n_kv_heads,
                        max_seq_len,
                        head_dim,
                    ),
                    dtype=dtype,
                    device=self.device,
                )
            )

        self.seq_lens = torch.zeros(
            max_batch_size,
            dtype=torch.int32,
            device=self.device,
        )

        self.active_batch_size = 0

        logger.info(
            "Allocated KV cache: layers=%d batch=%d seq=%d kv_heads=%d head_dim=%d memory=%.2f GiB",
            num_layers,
            max_batch_size,
            max_seq_len,
            n_kv_heads,
            head_dim,
            self.memory_footprint_bytes() / (1024**3),
        )

    @property
    def max_batch_size(self) -> int:
        return self.spec.max_batch_size

    @property
    def max_seq_len(self) -> int:
        return self.spec.max_seq_len

    @property
    def num_layers(self) -> int:
        return self.spec.num_layers

    @property
    def n_kv_heads(self) -> int:
        return self.spec.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.spec.head_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.spec.dtype

    def memory_footprint_bytes(self) -> int:
        total = 0

        for layer_cache in self.k_cache:
            total += layer_cache.numel() * layer_cache.element_size()

        for layer_cache in self.v_cache:
            total += layer_cache.numel() * layer_cache.element_size()

        return total

    def begin_request(
        self,
        batch_size: int,
    ) -> None:
        """
        Start a new static-batch request.

        No VRAM clearing is performed. Logical lengths determine validity.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size={batch_size} exceeds "
                f"max_batch_size={self.max_batch_size}"
            )

        self.active_batch_size = batch_size
        self.seq_lens[:batch_size].zero_()

    def reset(self) -> None:
        """
        Reset logical state without memset'ing the complete KV allocation.
        """
        if self.active_batch_size:
            self.seq_lens[: self.active_batch_size].zero_()

        self.active_batch_size = 0

    def current_seq_len(self, batch_idx: int = 0) -> int:
        if batch_idx < 0 or batch_idx >= self.max_batch_size:
            raise IndexError("invalid batch index")

        return int(self.seq_lens[batch_idx].item())

    @torch.no_grad()
    def update(
        self,
        *,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write one layer's K/V states into the corresponding cache.

        key_states/value_states:
            [B, KV_HEADS, SEQ, D]
        """
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx={layer_idx} outside [0, {self.num_layers})"
            )

        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError(
                "key_states and value_states must be [B, H, S, D]"
            )

        if key_states.shape != value_states.shape:
            raise ValueError("K/V shapes must match")

        batch_size, kv_heads, seq_len, head_dim = key_states.shape

        if kv_heads != self.n_kv_heads:
            raise ValueError(
                f"expected {self.n_kv_heads} KV heads, got {kv_heads}"
            )

        if head_dim != self.head_dim:
            raise ValueError(
                f"expected head_dim={self.head_dim}, got {head_dim}"
            )

        if batch_size > self.max_batch_size:
            raise ValueError("batch size exceeds cache capacity")

        if start_pos < 0:
            raise ValueError("start_pos must be non-negative")

        end_pos = start_pos + seq_len

        if end_pos > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: end_pos={end_pos}, "
                f"max_seq_len={self.max_seq_len}"
            )

        k_src = (
            key_states
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, self.n_kv_heads, self.head_dim)
        )

        v_src = (
            value_states
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, self.n_kv_heads, self.head_dim)
        )

        batch_offsets = (
            torch.arange(
                batch_size,
                device=key_states.device,
                dtype=torch.int64,
            )
            * self.max_seq_len
        ).unsqueeze(1)

        pos_offsets = torch.arange(
            start_pos,
            end_pos,
            device=key_states.device,
            dtype=torch.int64,
        ).unsqueeze(0)

        slot_mapping = (
            batch_offsets + pos_offsets
        ).reshape(-1)

        update_kv_cache(
            key_src=k_src,
            value_src=v_src,
            key_cache=self.k_cache[layer_idx],
            value_cache=self.v_cache[layer_idx],
            slot_mapping=slot_mapping,
        )

        self.seq_lens[:batch_size] = end_pos

        return (
            self.k_cache[layer_idx][
                :batch_size,
                :,
                :end_pos,
                :,
            ],
            self.v_cache[layer_idx][
                :batch_size,
                :,
                :end_pos,
                :,
            ],
        )

    def get(
        self,
        *,
        layer_idx: int,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError("invalid layer index")

        if batch_size > self.max_batch_size:
            raise ValueError("batch size exceeds cache capacity")

        if seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds cache capacity")

        return (
            self.k_cache[layer_idx][
                :batch_size,
                :,
                :seq_len,
                :,
            ],
            self.v_cache[layer_idx][
                :batch_size,
                :,
                :seq_len,
                :,
            ],
        )