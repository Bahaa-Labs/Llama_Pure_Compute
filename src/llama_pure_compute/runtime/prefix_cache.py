from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PrefixKey:
    digest: bytes


@dataclass(slots=True)
class PrefixEntry:
    key: PrefixKey
    token_count: int
    blocks: tuple[int, ...]


class PrefixCache:
    """
    Thread-safe LRU prefix-cache metadata.

    The cache stores references to already-computed KV blocks.
    Actual KV tensors remain owned by PagedKVCache.
    """

    def __init__(
        self,
        *,
        max_entries: int = 1024,
    ) -> None:
        self.max_entries = max_entries

        self._entries: OrderedDict[
            PrefixKey,
            PrefixEntry,
        ] = OrderedDict()

        self._lock = threading.RLock()

    @staticmethod
    def make_key(
        token_ids: list[int],
    ) -> PrefixKey:

        payload = b"".join(
            token.to_bytes(
                4,
                "little",
                signed=False,
            )
            for token in token_ids
        )

        return PrefixKey(
            hashlib.blake2b(
                payload,
                digest_size=16,
            ).digest()
        )

    def get(
        self,
        token_ids: list[int],
    ) -> PrefixEntry | None:

        key = self.make_key(
            token_ids
        )

        with self._lock:

            entry = self._entries.get(key)

            if entry is None:
                return None

            self._entries.move_to_end(
                key
            )

            return entry

    def put(
        self,
        token_ids: list[int],
        blocks: tuple[int, ...],
    ) -> None:

        key = self.make_key(
            token_ids
        )

        entry = PrefixEntry(
            key=key,
            token_count=len(token_ids),
            blocks=blocks,
        )

        with self._lock:

            self._entries[key] = entry
            self._entries.move_to_end(key)

            while (
                len(self._entries)
                > self.max_entries
            ):
                self._entries.popitem(
                    last=False
                )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)