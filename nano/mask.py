import torch


def _can_skip_sdpa_mask(
    attention_mask: torch.Tensor | None,
    q_len: int,
    kv_len: int,
) -> bool:
    if attention_mask is not None and attention_mask.dim() == 4:
        return False
    no_padding = attention_mask is None or bool(attention_mask.bool().all())
    return no_padding and (q_len == 1 or q_len == kv_len)


def _unmask_unattended(causal_mask: torch.Tensor, min_dtype: float) -> torch.Tensor:
    fully_masked = (causal_mask == min_dtype).all(dim=-1, keepdim=True)
    return causal_mask.masked_fill(fully_masked, 0.0)


def make_doc_causal_mask(document_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
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


def create_causal_mask(
    attention_mask: torch.Tensor | None,
    hidden_states: torch.Tensor,
    cache_position: torch.Tensor,
    kv_len: int,
    document_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask
    if document_ids is not None:
        return make_doc_causal_mask(document_ids, hidden_states.dtype)

    q_len = hidden_states.size(1)
    if _can_skip_sdpa_mask(attention_mask, q_len, kv_len):
        return None

    min_dtype = torch.finfo(hidden_states.dtype).min
    device = hidden_states.device
    causal = torch.full((q_len, kv_len), min_dtype, dtype=hidden_states.dtype, device=device)
    future = torch.arange(kv_len, device=device) > cache_position.reshape(-1, 1)
    causal *= future
    causal = causal[None, None, :, :].expand(hidden_states.size(0), 1, -1, -1)

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
