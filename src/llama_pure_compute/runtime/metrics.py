from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationMetrics:
    prompt_tokens: int
    generated_tokens: int
    total_latency_ms: float
    ttft_ms: float
    decode_latency_ms: float
    tokens_per_second: float

    @property
    def itl_ms(self) -> float:
        """Average inter-token latency excluding the first token."""
        if self.generated_tokens <= 1:
            return 0.0

        return self.decode_latency_ms / (
            self.generated_tokens - 1
        )