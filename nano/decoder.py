import torch
import torch.nn as nn

from .attention import LlamaAttention
from .cache import KVCache
from .config import NanoConfig
from .mlp import LlamaMLP
from .rmsnorm import LlamaRMSNorm


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: NanoConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
            cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states
