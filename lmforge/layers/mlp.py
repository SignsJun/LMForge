import torch
import torch.nn as nn
import torch.nn.functional as F

from lmforge.models.config import ModelConfig


class MLP(nn.Module):
    """SwiGLU：gate 和 up 两路，比普通 FFN 表达力更强，Llama/Qwen 都用这个。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
