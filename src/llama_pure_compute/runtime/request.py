from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence
import time


class RequestState(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class RequestLimits:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0


@dataclass(slots=True)
class InferenceRequest:
    request_id: str
    prompt_tokens: list[int]
    limits: RequestLimits

    state: RequestState = RequestState.QUEUED

    generated_tokens: list[int] = field(
        default_factory=list
    )

    sequence_position: int = 0
    prompt_position: int = 0

    created_at: float = field(
        default_factory=time.perf_counter
    )

    prefill_started_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    error: str | None = None

    cancelled: bool = False

    def total_tokens(self) -> int:
        return (
            self.prompt_position
            + len(self.generated_tokens)
        )

    def remaining_tokens(self) -> int:
        return max(
            self.limits.max_new_tokens
            - len(self.generated_tokens),
            0,
        )

    def mark_prefill(self) -> None:
        self.state = RequestState.PREFILL
        self.prefill_started_at = time.perf_counter()

    def mark_decode(self) -> None:
        self.state = RequestState.DECODE

    def mark_finished(self) -> None:
        self.state = RequestState.FINISHED
        self.finished_at = time.perf_counter()

    def mark_cancelled(self) -> None:
        self.cancelled = True
        self.state = RequestState.CANCELLED
        self.finished_at = time.perf_counter()

    def mark_failed(self, error: str) -> None:
        self.state = RequestState.FAILED
        self.error = error
        self.finished_at = time.perf_counter()

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None

        return (
            self.first_token_at
            - self.created_at
        ) * 1000.0

    @property
    def total_latency_ms(self) -> float | None:
        if self.finished_at is None:
            return None

        return (
            self.finished_at
            - self.created_at
        ) * 1000.0