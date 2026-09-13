from transformers import AutoTokenizer


class Tokenizer:
    """包一层 HF tokenizer。词表复用 Qwen，不自己训。"""

    def __init__(self, name_or_path: str = "Qwen/Qwen2.5-0.5B"):
        self.tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        # 有的词表没有 pad，用 eos 顶上，否则 collate 无法对齐
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id

    @property
    def eos_id(self) -> int:
        return self.tok.eos_token_id

    @property
    def bos_id(self) -> int | None:
        return self.tok.bos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # packing 时自己在文档末尾加 eos，避免 HF 偷偷插 bos
        return self.tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)
