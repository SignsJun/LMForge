from dataclasses import dataclass


@dataclass
class NanoConfig:
    vocab_size: int = 50257
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    attn_implementation: str = "sdpa"

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
