from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True, slots=True)
class BlockRef:
    block_id: int


class PagedKVCache:
    """
    GPU-resident block allocator for KV storage.

    Layout:
        [num_blocks, kv_heads, block_size, head_dim]

    Physical blocks are independent of logical request positions.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )

        self.k_cache = [
            torch.empty(
                shape,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

        self.v_cache = [
            torch.empty(
                shape,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

        self._free = list(
            reversed(
                range(num_blocks)
            )
        )

        self._request_blocks: dict[
            str,
            list[int],
        ] = {}

    def allocate(
        self,
        request_id: str,
        count: int,
    ) -> list[BlockRef]:

        if count <= 0:
            return []

        if len(self._free) < count:
            raise RuntimeError(
                "KV cache capacity exhausted."
            )

        if request_id in self._request_blocks:
            raise RuntimeError(
                "Request already owns cache blocks."
            )

        blocks = [
            self._free.pop()
            for _ in range(count)
        ]

        self._request_blocks[
            request_id
        ] = blocks

        return [
            BlockRef(block)
            for block in blocks
        ]

    def grow(
        self,
        request_id: str,
        count: int,
    ) -> list[BlockRef]:

        if count <= 0:
            return []

        blocks = self._request_blocks.get(
            request_id
        )

        if blocks is None:
            raise KeyError(request_id)

        if len(self._free) < count:
            raise RuntimeError(
                "KV cache capacity exhausted."
            )

        new_blocks = [
            self._free.pop()
            for _ in range(count)
        ]

        blocks.extend(new_blocks)

        return [
            BlockRef(block)
            for block in new_blocks
        ]

    def release(
        self,
        request_id: str,
    ) -> None:

        blocks = self._request_blocks.pop(
            request_id,
            None,
        )

        if not blocks:
            return

        self._free.extend(blocks)

    def blocks_for(
        self,
        request_id: str,
    ) -> tuple[BlockRef, ...]:

        blocks = self._request_blocks.get(
            request_id
        )

        if blocks is None:
            raise KeyError(request_id)

        return tuple(
            BlockRef(block)
            for block in blocks
        )

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return (
            self.num_blocks
            - len(self._free)
        )

    @property
    def utilization(self) -> float:
        if self.num_blocks == 0:
            return 0.0

        return (
            self.used_blocks
            / self.num_blocks
        )