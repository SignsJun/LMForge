import torch
import torch.nn as nn

from lmforge.layers.attention import Attention
from lmforge.layers.cache import KVCache
from lmforge.layers.mlp import MLP
from lmforge.layers.rmsnorm import RMSNorm
from lmforge.models.config import ModelConfig


class DecoderLayer(nn.Module):
    """Pre-norm：先 RMSNorm 再子层，残差在子层外相加，训练更稳。"""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention_mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
