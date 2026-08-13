from __future__ import annotations

import asyncio
import logging
from collections import deque

from llama_pure_compute.runtime.request import (
    InferenceRequest,
    RequestState,
)

logger = logging.getLogger(
    "llama_pure_compute.scheduler"
)


class Scheduler:
    """
    Async continuous-batching scheduler.

    Responsibilities:
        - admission control
        - FIFO queueing
        - cancellation
        - active request tracking
        - decode batch selection
        - backpressure
    """

    def __init__(
        self,
        *,
        max_queue_size: int = 256,
        max_active_requests: int = 32,
    ) -> None:
        self.max_queue_size = max_queue_size
        self.max_active_requests = max_active_requests

        self._queue: deque[
            InferenceRequest
        ] = deque()

        self._active: dict[
            str,
            InferenceRequest,
        ] = {}

        self._lock = asyncio.Lock()

    async def submit(
        self,
        request: InferenceRequest,
    ) -> None:

        async with self._lock:

            if request.request_id in self._active:
                raise ValueError(
                    f"Duplicate request ID: "
                    f"{request.request_id}"
                )

            if (
                len(self._queue)
                >= self.max_queue_size
            ):
                raise RuntimeError(
                    "Scheduler queue is full."
                )

            if (
                len(self._active)
                >= self.max_active_requests
            ):
                raise RuntimeError(
                    "Maximum active request "
                    "capacity reached."
                )

            self._queue.append(request)
            self._active[
                request.request_id
            ] = request

    async def admit(
        self,
    ) -> list[InferenceRequest]:

        async with self._lock:

            admitted: list[
                InferenceRequest
            ] = []

            while (
                self._queue
                and len(admitted)
                < self.max_active_requests
            ):
                request = self._queue.popleft()

                if request.cancelled:
                    request.mark_cancelled()
                    continue

                admitted.append(request)

            return admitted

    async def cancel(
        self,
        request_id: str,
    ) -> bool:

        async with self._lock:

            request = self._active.get(
                request_id
            )

            if request is None:
                return False

            request.mark_cancelled()

            return True

    async def finish(
        self,
        request_id: str,
    ) -> None:

        async with self._lock:
            self._active.pop(
                request_id,
                None,
            )

    async def get(
        self,
        request_id: str,
    ) -> InferenceRequest | None:

        async with self._lock:
            return self._active.get(
                request_id
            )

    async def active_requests(
        self,
    ) -> list[InferenceRequest]:

        async with self._lock:
            return list(
                self._active.values()
            )

    async def decode_batch(
        self,
    ) -> list[InferenceRequest]:

        async with self._lock:

            batch = []

            for request in self._active.values():

                if request.cancelled:
                    continue

                if request.state not in (
                    RequestState.PREFILL,
                    RequestState.DECODE,
                ):
                    continue

                if request.remaining_tokens() <= 0:
                    continue

                batch.append(request)

            return batch