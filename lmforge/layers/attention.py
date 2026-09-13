import torch
import torch.nn as nn
import torch.nn.functional as F

from lmforge.layers.cache import KVCache
from lmforge.layers.rope import apply_rotary_pos_emb
from lmforge.models.config import ModelConfig


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA：把 n_kv_heads 份 K/V 扩成 n_heads 份，让每个 Q head 都有对应的 KV。

    缓存里只存未展开的 K/V，这里再 expand，省显存。
    """
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(bsz, n_kv, n_rep, slen, head_dim)
        .reshape(bsz, n_kv * n_rep, slen, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_key_value_groups
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        # RoPE 打在“当前步”的 Q/K 上，cache 里的旧 K 已经带过位置编码，不能再打一遍
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        if attention_mask is not None:
            attention_mask = attention_mask[..., : k.size(2)].to(dtype=q.dtype)
        # mask 已在模型 forward 合成。None 且等长 → is_causal 走 Flash；decode 不等长则关 is_causal
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None and q_len == k.size(2),
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(out)
