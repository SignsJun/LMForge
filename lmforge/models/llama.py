import torch
import torch.nn as nn
import torch.nn.functional as F

from lmforge.layers.cache import KVCache
from lmforge.layers.decoder import DecoderLayer
from lmforge.layers.mask import create_causal_mask
from lmforge.layers.rmsnorm import RMSNorm
from lmforge.layers.rope import RotaryEmbedding
from lmforge.models.config import ModelConfig


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            # 输入输出共享词表矩阵，少一套 embedding 参数
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = input_ids.shape
        past_seen = 0 if cache is None else cache.get_seq_length()
        # 与 Qwen 的 cache_position 相同：当前这批 token 在整段序列里的绝对下标
        cache_position = torch.arange(past_seen, past_seen + q_len, device=input_ids.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(bsz, -1)

        hidden = self.embed_tokens(input_ids)
        kv_len = past_seen + q_len
        # Qwen3：causal + padding 在模型侧合成，Attention 不再自己画 mask
        attention_mask = create_causal_mask(
            attention_mask,
            hidden,
            cache_position,
            kv_len,
            document_ids,
        )
        cos, sin = self.rotary_emb(hidden, position_ids)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, attention_mask, cache)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # logits[t] 预测 labels[t+1]；ignore_index=-100 跳过 pad / 文档首 token
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        cache = KVCache()
        generated = input_ids
        for _ in range(max_new_tokens):
            # 第一步跑完整 prompt 写入 cache，之后每步只进 1 个 token
            step_ids = generated if cache.get_seq_length() == 0 else generated[:, -1:]
            logits, _ = self(step_ids, cache=cache)
            next_logits = logits[:, -1]
            if temperature <= 0:
                next_tokens = next_logits.argmax(-1, keepdim=True)
            else:
                next_tokens = torch.multinomial(torch.softmax(next_logits / temperature, dim=-1), 1)
            generated = torch.cat([generated, next_tokens], dim=1)
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        return generated
