import torch


def _can_skip_sdpa_mask(
    attention_mask: torch.Tensor | None,
    q_len: int,
    kv_len: int,
) -> bool:
    """没有 pad、且 SDPA 能自己处理因果时，返回 None 走 Flash。

    对齐 Qwen3 `_ignore_causal_mask_sdpa`：
    - prefill：q_len == kv_len，用 is_causal
    - decode：q_len == 1，单 query 看全部 KV，不用 4D mask
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        return False
    no_padding = attention_mask is None or bool(attention_mask.bool().all())
    return no_padding and (q_len == 1 or q_len == kv_len)


def _unmask_unattended(causal_mask: torch.Tensor, min_dtype: float) -> torch.Tensor:
    """左 padding 时 pad 行整行 -inf，SDPA softmax 会 NaN，把这些行改成 0。"""
    fully_masked = (causal_mask == min_dtype).all(dim=-1, keepdim=True)
    return causal_mask.masked_fill(fully_masked, 0.0)


def create_causal_mask(
    attention_mask: torch.Tensor | None,
    hidden_states: torch.Tensor,
    cache_position: torch.Tensor,
    kv_len: int,
    document_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Qwen3 风格：在 Attention 外把 causal 和 padding 合成一份。

    Qwen3 的 `create_causal_mask` 不在 Attention 里画 mask，而是模型 forward
    里先做好，再传给每一层。Attention 只负责：有 mask 就用，没有就 is_causal。

    - `attention_mask` 2D：`1` 有效、`0` pad
    - 已是 4D：视为已经合成过，直接返回
    - `document_ids`：packed 额外约束，HF 用 packed_sequence_mask，我们单独合成
    - 返回 None：没有 pad，让 SDPA 走 Flash
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    if document_ids is not None:
        return make_doc_causal_mask(document_ids, hidden_states.dtype)

    q_len = hidden_states.size(1)
    if _can_skip_sdpa_mask(attention_mask, q_len, kv_len):
        return None

    # 1) causal：key 下标 > query 绝对位置 => 未来 token
    #    decode 时 q_len=1、kv_len>>1，必须用 cache_position，不能 tril
    min_dtype = torch.finfo(hidden_states.dtype).min
    device = hidden_states.device
    causal = torch.full((q_len, kv_len), min_dtype, dtype=hidden_states.dtype, device=device)
    future = torch.arange(kv_len, device=device) > cache_position.reshape(-1, 1)
    causal *= future
    causal = causal[None, None, :, :].expand(hidden_states.size(0), 1, -1, -1)

    # 2) 叠 padding：1=可见、0=pad。causal 可见处是 0，加上 pad 的 0 仍为 0 → 挡住
    if attention_mask is not None:
        causal = causal.clone()
        if attention_mask.shape[-1] > kv_len:
            attention_mask = attention_mask[:, :kv_len]
        pad_len = attention_mask.shape[-1]
        blocked = causal[:, :, :, :pad_len] + attention_mask[:, None, None, :].to(
            device=causal.device, dtype=causal.dtype
        )
        causal[:, :, :, :pad_len] = causal[:, :, :, :pad_len].masked_fill(blocked == 0, min_dtype)

    return _unmask_unattended(causal, min_dtype)


def make_doc_causal_mask(document_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """packed pretrain：同文档因果可见，跨文档切断。Qwen3 默认一条序列同一篇文档。"""
    bsz, seqlen = document_ids.shape
    device = document_ids.device
    causal = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device).tril()
    valid = document_ids >= 0
    same_doc = document_ids.unsqueeze(2) == document_ids.unsqueeze(1)
    allowed = causal & same_doc & valid.unsqueeze(1) & valid.unsqueeze(2)
    eye = torch.eye(seqlen, dtype=torch.bool, device=device)
    allowed = allowed | (~valid).unsqueeze(2) & eye
    mask = torch.zeros(bsz, 1, seqlen, seqlen, dtype=dtype, device=device)
    mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return mask
