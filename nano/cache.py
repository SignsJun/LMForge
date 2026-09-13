import torch


class KVCache:
    def __init__(self):
        self.k: list[torch.Tensor] = []
        self.v: list[torch.Tensor] = []

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == len(self.k):
            self.k.append(key)
            self.v.append(value)
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], key], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], value], dim=2)
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self) -> int:
        return 0 if not self.k else self.k[0].size(2)
