import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVCache
from .config import NanoConfig
from .decoder import LlamaDecoderLayer
from .rmsnorm import LlamaRMSNorm
from .rope import LlamaRotaryEmbedding


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config.head_dim, config.rope_theta)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
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
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, KVCache | None]:
        bsz, q_len = input_ids.shape
        if use_cache and cache is None:
            cache = KVCache()
        past_seen_tokens = 0 if cache is None else cache.get_seq_length()
        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + q_len, device=input_ids.device
            ).unsqueeze(0).expand(bsz, -1)
        if (
            attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.size(1) == q_len
            and past_seen_tokens
        ):
            attention_mask = torch.cat(
                [attention_mask.new_ones(bsz, past_seen_tokens), attention_mask], dim=1
            )

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                cache,
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return logits, loss, cache

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
            step_ids = generated if cache.get_seq_length() == 0 else generated[:, -1:]
            logits, _, cache = self(step_ids, cache=cache, use_cache=True)
            next_logits = logits[:, -1]
            if temperature <= 0:
                next_tokens = next_logits.argmax(-1, keepdim=True)
            else:
                next_tokens = torch.multinomial(torch.softmax(next_logits / temperature, dim=-1), 1)
            generated = torch.cat([generated, next_tokens], dim=1)
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        return generated


NanoLM = LlamaForCausalLM
