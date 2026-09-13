from lmforge.models.config import ModelConfig

__all__ = ["KVCache", "ModelConfig", "LlamaForCausalLM"]


def __getattr__(name: str):
    if name == "KVCache":
        from lmforge.layers.cache import KVCache

        return KVCache
    if name == "LlamaForCausalLM":
        from lmforge.models.llama import LlamaForCausalLM

        return LlamaForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
