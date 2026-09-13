import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer: Optimizer, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> LambdaLR:
    """线性 warmup 再 cosine 降到 min_lr。LambdaLR 的倍数是相对初始 lr 的。"""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + coeff * (lr - min_lr)) / lr

    return LambdaLR(optimizer, lr_lambda)
