from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    
    def byte_size(self) -> float:
        sizes = {
            Precision.FP32: 4.0,
            Precision.FP16: 2.0,
            Precision.BF16: 2.0,
            Precision.INT8: 1.0,
            Precision.INT4: 0.5,
        }
        return sizes[self]

@dataclass
class LlamaModelConfig:
    # Model Architecture Parameters
    vocab_size: int = 32000
    dim: int = 4096
    inter_dim: int = 11008
    num_layers: int = 32 
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # Default to MHA if None
    head_dim: int = field(init=False)
    
    # Sequence and Batch Specs
    max_seq_len: int = 4096
    max_batch_size: int = 1
    
    # Rotary Embeddings and LayerNorm Specs
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Execution and Memory Settings
    precision: Precision = Precision.FP16
    aligned_byte_boundary: int = 64
    
    def __post_init__(self) -> None:
        # Handle multi-query / grouped-query attention default
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        
        # Auto-compute head dimension
        if self.dim % self.num_heads != 0:
            raise ValueError(f"Dim ({self.dim}) must be divisible by num_heads ({self.num_heads})")
        
        self.head_dim = self.dim // self.num_heads
        self._validate()

    def _validate(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be multiple of num_kv_heads ({self.num_kv_heads})"
            )
        if self.max_seq_len <= 0 or self.max_batch_size <= 0:
            raise ValueError("Sequence lengths and batch sizes must be strictly positive")

    @property
    def gqa_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_cache_elements_per_token(self) -> int:
        return 2 * self.num_kv_heads * self.head_dim

    def estimate_kv_cache_memory_bytes(self) -> int:
        total_elements = (
            self.num_layers * self.max_batch_size * self.max_seq_len *
            self.kv_cache_elements_per_token
        )
        return int(total_elements * self.precision.byte_size())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LlamaModelConfig:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values() if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        if "precision" in filtered_data and isinstance(filtered_data["precision"], str):
            filtered_data["precision"] = Precision(filtered_data["precision"].lower())
        return cls(**filtered_data)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> LlamaModelConfig:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        key_mapping = {
            "hidden_size": "dim",
            "intermediate_size": "inter_dim",
            "num_hidden_layers": "num_layers",
            "num_attention_heads": "num_heads",  
            "num_key_value_heads": "num_kv_heads",
            "rms_norm_eps": "rms_norm_eps",
            "rope_theta": "rope_theta",
        }
        
        normalized_data = {}
        for k, v in data.items():
            mapped_key = key_mapping.get(k, k)
            normalized_data[mapped_key] = v
        
        return cls.from_dict(normalized_data)

    @classmethod
    def from_env(cls) -> LlamaModelConfig:
        env_config = {}
        if "LLAMA_MAX_BATCH_SIZE" in os.environ:
            env_config["max_batch_size"] = int(os.environ["LLAMA_MAX_BATCH_SIZE"])
        if "LLAMA_MAX_SEQ_LEN" in os.environ:
            env_config["max_seq_len"] = int(os.environ["LLAMA_MAX_SEQ_LEN"])
        if "LLAMA_PRECISION" in os.environ:
            env_config["precision"] = Precision(os.environ["LLAMA_PRECISION"].lower())
        
        return cls.from_dict(env_config)

    def to_json(self, output_path: Union[str, Path]) -> None:
        data = asdict(self)
        data["precision"] = self.precision.value
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def generate_cpp_header(self, output_path: Union[str, Path]) -> None:
        content = f"""// Auto-generated header for Llama_Pure_Compute Runtime Configuration
#ifndef LLAMA_CONFIG_AUTO_H
#define LLAMA_CONFIG_AUTO_H

#include <cstddef>

namespace llama_pure_compute {{

struct StaticModelConfig {{
    static constexpr size_t VOCAB_SIZE = {self.vocab_size};
    static constexpr size_t DIM = {self.dim};
    static constexpr size_t INTER_DIM = {self.inter_dim};
    static constexpr size_t NUM_LAYERS = {self.num_layers};
    static constexpr size_t NUM_HEADS = {self.num_heads};
    static constexpr size_t NUM_KV_HEADS = {self.num_kv_heads};
    static constexpr size_t HEAD_DIM = {self.head_dim};
    static constexpr size_t MAX_SEQ_LEN = {self.max_seq_len};
    static constexpr size_t MAX_BATCH_SIZE = {self.max_batch_size};
    static constexpr float RMS_NORM_EPS = {self.rms_norm_eps}f;
    static constexpr float ROPE_THETA = {self.rope_theta}f;
    static constexpr size_t ALIGNED_BYTES = {self.aligned_byte_boundary};
}};

}} // namespace llama_pure_compute
#endif // LLAMA_CONFIG_AUTO_H
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)


# Presets for Standard Llama model variants
PRESETS: Dict[str, LlamaModelConfig] = {
    "llama-2-7b": LlamaModelConfig(
        vocab_size=32000, dim=4096, inter_dim=11008, num_layers=32, num_heads=32, num_kv_heads=32
    ),
    "llama-3-8b": LlamaModelConfig(
        vocab_size=128256, dim=4096, inter_dim=14336, num_layers=32, num_heads=32, num_kv_heads=8, rope_theta=500000.0
    ),
    "llama-2-70b": LlamaModelConfig(
        vocab_size=32000, dim=8192, inter_dim=28672, num_layers=80, num_heads=64, num_kv_heads=8
    ),
}

# Example Usage
if __name__ == "__main__":
    config = PRESETS["llama-3-8b"]
    config.max_batch_size = 4
    config.max_seq_len = 8192
    
    print("Llama_Pure_Compute Config")
    print("Model Architecture: Llama 3 8B")
    print(f"Dimension: {config.dim}, Head Dim: {config.head_dim}")
    print(f"Grouped Query Attention (GQA) Ratio: {config.gqa_group_size}: 1")
    print(f"Estimated KV-Cache Size: {config.estimate_kv_cache_memory_bytes() / (1024**3):.2f} GB")