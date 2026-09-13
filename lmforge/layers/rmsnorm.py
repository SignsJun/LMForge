import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """只做均方根缩放，没有均值中心化，比 LayerNorm 省、也是 Llama/Qwen 的默认。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(orig_dtype)
