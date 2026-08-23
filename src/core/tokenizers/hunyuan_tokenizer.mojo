# core/tokenizers/hunyuan_tokenizer.mojo
#
# Hunyuan-family tokenizer (hunyuan-dense / hunyuan-large, e.g. Hy-MT2).
# GPT-2 byte-level BPE with the hunyuan special-token table
# (`<｜start▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, `<|extra_N|>`, ...).
# When no tokenizer.json is present the added tokens are derived from the
# GGUF vocabulary (special-token heuristic, same family as llama.cpp's
# `llm_tokenizer_bpe::init`).

from .bpe_engine import (
    BpeTokenizer,
    AddedToken,
    FLAVOR_HUNYUAN,
    read_gguf_token_list,
)
from ..gguf_loader import GGUFContext
from ..tokenizer import Tokenizer


struct HunyuanTokenizer(Movable, Tokenizer):
    var _engine: BpeTokenizer

    def __init__(out self):
        self._engine = BpeTokenizer()

    @staticmethod
    def load(tokenizer_json_path: String, ctx: GGUFContext) raises -> Self:
        var tokenizer = Self()
        tokenizer._engine = BpeTokenizer.load_flavor(
            tokenizer_json_path, ctx, FLAVOR_HUNYUAN
        )
        return tokenizer^

    @staticmethod
    def load_from_gguf(ctx: GGUFContext) raises -> Self:
        var tokenizer = Self()
        var engine = BpeTokenizer.load_from_gguf(ctx)
        engine._flavor = FLAVOR_HUNYUAN
        engine.derive_added_tokens()
        tokenizer._engine = engine^
        return tokenizer^

    def encode(self, text: String) -> List[Int]:
        return self._engine.encode(text)

    def encode_with_bos(self, text: String) -> List[Int]:
        return self._engine.encode_with_bos(text)

    def decode(self, tokens: List[Int]) -> String:
        return self._engine.decode(tokens)

    def vocab_size(self) -> Int:
        return self._engine.vocab_size()

    def bos_id(self) -> Int:
        return self._engine.bos_id()

    def eos_id(self) -> Int:
        return self._engine.eos_id()

    def engine(self) -> BpeTokenizer:
        return self._engine
