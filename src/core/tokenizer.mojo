# core/tokenizer.mojo
#
# M7: the tokenizer interface contract plus the flavor tags shared by every
# tokenizer implementation.
#
# `Tokenizer` is the trait every tokenizer must conform to (encode / decode /
# vocab_size / bos_id / eos_id / encode_with_bos).  The concrete engine for
# every BPE family is the generic `BpeTokenizer` in
# core/tokenizers/bpe_engine.mojo; model families differ only in data (the
# flavor tag, the added-token table and the bos/eos ids), selected from the
# GGUF metadata by `make_tokenizer` (core/tokenizers/registry.mojo).
#
# Mojo 1.0 note: this toolchain has no usable trait-existential dynamic
# dispatch (trait-typed values box to `AnyTrait`, but method calls on the
# box fail to bind `self`) and no runtime mutable globals, so the
# *runtime* dispatch happens through the flavor tag inside `BpeTokenizer`,
# while the trait serves as the static contract for generic code.  Custom
# tokenizers are registered through `TokenizerRegistry` /
# `register_tokenizer` (core/tokenizers/registry.mojo) with a data-driven
# spec - vocab/merges come from the GGUF metadata, the spec supplies the
# added-token table, flavor and boundary ids.


trait Tokenizer:
    def encode(self, text: String) -> List[Int]:
        ...

    def decode(self, tokens: List[Int]) -> String:
        ...

    def vocab_size(self) -> Int:
        ...

    def bos_id(self) -> Int:
        ...

    def eos_id(self) -> Int:
        ...

    def encode_with_bos(self, text: String) -> List[Int]:
        ...


# -- flavor tags (struct-register style, see device.mojo) ---------------------


struct TokenizerFlavor(Copyable, Equatable, ImplicitlyCopyable, Movable):
    var _tag: Int8

    def __init__(out self, tag: Int8):
        self._tag = tag

    comptime Qwen = TokenizerFlavor(Int8(0))
    comptime Llama = TokenizerFlavor(Int8(1))
    comptime Hunyuan = TokenizerFlavor(Int8(2))
    comptime Gpt2 = TokenizerFlavor(Int8(3))
    comptime Custom = TokenizerFlavor(Int8(4))

    def __eq__(self, other: Self) -> Bool:
        return self._tag == other._tag

    def __ne__(self, other: Self) -> Bool:
        return self._tag != other._tag


def flavor_name(flavor: TokenizerFlavor) -> String:
    if flavor == TokenizerFlavor.Qwen:
        return String("qwen")
    if flavor == TokenizerFlavor.Llama:
        return String("llama")
    if flavor == TokenizerFlavor.Hunyuan:
        return String("hunyuan")
    if flavor == TokenizerFlavor.Custom:
        return String("custom")
    return String("gpt2")


def flavor_tag(flavor: TokenizerFlavor) -> Int8:
    return flavor._tag


# -- placeholder (kept for M2-era callers / import compatibility) -------------


struct DummyTokenizer(Tokenizer):
    var _vocab_size: Int

    def __init__(out self, vocab_size: Int = 151936):
        self._vocab_size = vocab_size

    def encode(self, text: String) -> List[Int]:
        """Return a hardcoded token sequence (M2 placeholder)."""
        _ = text
        var tokens = List[Int]()
        for i in range(5):
            tokens.append(i + 1)
        return tokens^

    def decode(self, tokens: List[Int]) -> String:
        """Return a hardcoded string (M2 placeholder)."""
        _ = tokens
        return String("placeholder output")

    def vocab_size(self) -> Int:
        return self._vocab_size

    def bos_id(self) -> Int:
        return 1

    def eos_id(self) -> Int:
        return 2

    def encode_with_bos(self, text: String) -> List[Int]:
        var tokens = self.encode(text)
        tokens.insert(0, self.bos_id())
        return tokens^
