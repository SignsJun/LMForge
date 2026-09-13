import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVCache
from .config import NanoConfig
from .rope import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(bsz, n_kv, n_rep, slen, head_dim)
        .reshape(bsz, n_kv * n_rep, slen, head_dim)
    )


def eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    training: bool,
) -> torch.Tensor:
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)
    return torch.matmul(attn_weights, value_states)


class LlamaAttention(nn.Module):
    def __init__(self, config: NanoConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim ** -0.5
        self.attn_implementation = config.attn_implementation

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _build_mask(
        self,
        attention_mask: torch.Tensor | None,
        q_len: int,
        kv_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None or bool(attention_mask.bool().all()):
            return None
        cache_position = torch.arange(kv_len - q_len, kv_len, device=device)
        kv_position = torch.arange(kv_len, device=device)
        causal = kv_position.unsqueeze(0) <= cache_position.unsqueeze(1)
        allowed = causal.view(1, 1, q_len, kv_len) & attention_mask[:, None, None, :].bool()
        mask = torch.zeros(batch_size, 1, q_len, kv_len, dtype=dtype, device=device)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        kv_len = key_states.size(2)
        attn_mask = self._build_mask(
            attention_mask, q_len, kv_len, bsz, query_states.dtype, query_states.device
        )
        is_causal = attn_mask is None and q_len == kv_len

        if self.attn_implementation == "sdpa":
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            if attn_mask is None:
                cache_position = torch.arange(kv_len - q_len, kv_len, device=query_states.device)
                kv_position = torch.arange(kv_len, device=query_states.device)
                causal = kv_position.unsqueeze(0) <= cache_position.unsqueeze(1)
                attn_mask = torch.zeros(
                    bsz, 1, q_len, kv_len, dtype=query_states.dtype, device=query_states.device
                )
                attn_mask.masked_fill_(
                    ~causal.view(1, 1, q_len, kv_len), torch.finfo(query_states.dtype).min
                )
            attn_output = eager_attention_forward(
                query_states,
                key_states,
                value_states,
                attn_mask,
                self.attention_dropout if self.training else 0.0,
                self.scaling,
                self.training,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)
