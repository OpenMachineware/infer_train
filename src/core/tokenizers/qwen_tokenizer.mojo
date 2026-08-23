# core/tokenizers/qwen_tokenizer.mojo
#
# Qwen-family tokenizer (Qwen2 / Qwen2.5 / Qwen3 / DeepSeek-R1 distilled
# Qwen).  GPT-2 byte-level BPE plus the Qwen added-token table
# (`<｜User｜>`, `<think>`, `</think>`, ...) which is pre-split before BPE so
# the chat/think markers tokenize as single ids.

from .bpe_engine import (
    BpeTokenizer,
    AddedToken,
    FLAVOR_QWEN,
    read_gguf_token_list,
)
from ..gguf_loader import GGUFContext, get_meta_uint
from ..tokenizer import Tokenizer


struct QwenTokenizer(Movable, Tokenizer):
    var _engine: BpeTokenizer

    def __init__(out self):
        self._engine = BpeTokenizer()

    @staticmethod
    def load(tokenizer_json_path: String, ctx: GGUFContext) raises -> Self:
        """Load from `tokenizer.json` (vocab/merges/added tokens) + GGUF ids."""
        var tokenizer = Self()
        tokenizer._engine = BpeTokenizer.load_flavor(
            tokenizer_json_path, ctx, FLAVOR_QWEN
        )
        return tokenizer^

    @staticmethod
    def load_from_gguf(ctx: GGUFContext) raises -> Self:
        """Build from the GGUF vocabulary alone (no tokenizer.json)."""
        var tokenizer = Self()
        var engine = BpeTokenizer.load_from_gguf(ctx)
        engine._flavor = FLAVOR_QWEN
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
