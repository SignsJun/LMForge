import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(2).to(dtype=x.dtype)
        sin = emb.sin().unsqueeze(2).to(dtype=x.dtype)
        return cos, sin
