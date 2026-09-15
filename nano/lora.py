import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("lora r must be > 0")
        # 冻结base（原始模型）的权重
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        # 创建LoRA的low-rank矩阵A和B
        self.lora_A = nn.Parameter(torch.empty(base.in_features, r))
        self.lora_B = nn.Parameter(torch.empty(r, base.out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    dropout: float = 0.0,
) -> nn.Module:
    for name, module in list(model.named_modules()):
        child = name.rsplit(".", 1)[-1]
        if child not in target_modules or not isinstance(module, nn.Linear):
            continue
        parent_path = name.rsplit(".", 1)[0] if "." in name else ""
        parent = model if not parent_path else model.get_submodule(parent_path)
        setattr(parent, child, LoRALinear(module, r, alpha, dropout))
    freeze_non_lora(model)
    return model


def freeze_non_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def lora_param_stats(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def has_lora_state(state: dict) -> bool:
    return any("lora_A" in k for k in state)


def load_base_weights(model: nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
