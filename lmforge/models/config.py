from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    # 小于 num_attention_heads 就是 GQA；相等则退化为 MHA
    num_key_value_heads: int = 12
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
