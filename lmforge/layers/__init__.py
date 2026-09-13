__all__ = [
    "Attention",
    "DecoderLayer",
    "KVCache",
    "MLP",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "make_doc_causal_mask",
    "create_causal_mask",
]


def __getattr__(name: str):
    if name == "Attention":
        from lmforge.layers.attention import Attention

        return Attention
    if name == "DecoderLayer":
        from lmforge.layers.decoder import DecoderLayer

        return DecoderLayer
    if name == "KVCache":
        from lmforge.layers.cache import KVCache

        return KVCache
    if name == "MLP":
        from lmforge.layers.mlp import MLP

        return MLP
    if name == "RMSNorm":
        from lmforge.layers.rmsnorm import RMSNorm

        return RMSNorm
    if name == "RotaryEmbedding":
        from lmforge.layers.rope import RotaryEmbedding

        return RotaryEmbedding
    if name == "apply_rotary_pos_emb":
        from lmforge.layers.rope import apply_rotary_pos_emb

        return apply_rotary_pos_emb
    if name == "make_doc_causal_mask":
        from lmforge.layers.mask import make_doc_causal_mask

        return make_doc_causal_mask
    if name == "create_causal_mask":
        from lmforge.layers.mask import create_causal_mask

        return create_causal_mask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
