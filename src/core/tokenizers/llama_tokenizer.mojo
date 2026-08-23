# core/tokenizers/llama_tokenizer.mojo
#
# Llama-family tokenizer (llama-bpe / smaug-bpe / olmo / vicuna pretokenizers).
# Plain GPT-2 byte-level BPE: no added-token pre-splitting, bos/eos from the
# GGUF metadata (defaults 1 / 2).

from .bpe_engine import (
    BpeTokenizer,
    AddedToken,
    FLAVOR_LLAMA,
    read_gguf_token_list,
)
from ..gguf_loader import GGUFContext
from ..tokenizer import Tokenizer


struct LlamaTokenizer(Movable, Tokenizer):
    var _engine: BpeTokenizer

    def __init__(out self):
        self._engine = BpeTokenizer()

    @staticmethod
    def load(tokenizer_json_path: String, ctx: GGUFContext) raises -> Self:
        var tokenizer = Self()
        tokenizer._engine = BpeTokenizer.load_flavor(
            tokenizer_json_path, ctx, FLAVOR_LLAMA
        )
        return tokenizer^

    @staticmethod
    def load_from_gguf(ctx: GGUFContext) raises -> Self:
        var tokenizer = Self()
        var engine = BpeTokenizer.load_from_gguf(ctx)
        engine._flavor = FLAVOR_LLAMA
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
