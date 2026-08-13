from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


REQUESTS_TOTAL = Counter(
    "llama_requests_total",
    "Total inference requests.",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "llama_request_latency_seconds",
    "End-to-end request latency.",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
    ),
)

TTFT = Histogram(
    "llama_ttft_seconds",
    "Time to first generated token.",
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
    ),
)

DECODE_ITL = Histogram(
    "llama_decode_inter_token_seconds",
    "Inter-token latency.",
    buckets=(
        0.001,
        0.002,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
)

TOKENS_GENERATED = Counter(
    "llama_generated_tokens_total",
    "Number of generated tokens.",
)

ACTIVE_REQUESTS = Gauge(
    "llama_active_requests",
    "Currently active requests.",
)

KV_UTILIZATION = Gauge(
    "llama_kv_cache_utilization",
    "KV cache utilization.",
)