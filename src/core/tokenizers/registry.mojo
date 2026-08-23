# core/tokenizers/registry.mojo
#
# M7: tokenizer auto-selection and the custom-tokenizer registry.
#
# `make_tokenizer(ctx, tokenizer_json_path)` reads the GGUF tokenizer
# metadata (`tokenizer.ggml.model` / `tokenizer.ggml.pre`) and returns the
# matching flavor engine:
#
#   pre/model          -> flavor            (struct that owns the config)
#   qwen*, deepseek*   -> QwenTokenizer
#   hunyuan*           -> HunyuanTokenizer
#   llama*/smaug*/...  -> LlamaTokenizer
#   (registry hit)     -> user-registered TokenizerSpec
#   gpt2               -> LlamaTokenizer (plain GPT-2 BPE fallback)
#
# `register_tokenizer(registry, name, spec)` adds a custom tokenizer.  The
# vocab and merge table always come from the GGUF metadata
# (`tokenizer.ggml.tokens` / `tokenizer.ggml.merges`); the spec supplies the
# added-token pre-split table, the flavor tag and the boundary ids.
#
# Mojo 1.0 note: runtime mutable globals do not exist and trait-existential
# dispatch is not usable in this toolchain, so the registry is an explicit
# value (`TokenizerRegistry`) instead of a hidden global - it can be threaded
# through `load_model` / the server, or kept in the caller's scope.

from .bpe_engine import (
    BpeTokenizer,
    AddedToken,
    detect_flavor,
    read_gguf_token_list,
    read_gguf_merge_list,
    FLAVOR_QWEN,
    FLAVOR_LLAMA,
    FLAVOR_HUNYUAN,
    FLAVOR_GPT2,
    FLAVOR_CUSTOM,
)
from .qwen_tokenizer import QwenTokenizer
from .llama_tokenizer import LlamaTokenizer
from .hunyuan_tokenizer import HunyuanTokenizer
from ..gguf_loader import (
    GGUFContext,
    get_meta_uint,
    get_meta_str,
)
from ..tokenizer import Tokenizer, TokenizerFlavor, flavor_tag


struct TokenizerSpec(Copyable, Movable):
    """Data-driven description of a custom tokenizer.

    `name` is matched against the GGUF's `tokenizer.ggml.model` value (or
    `general.architecture` when the former is absent).  `added` is the
    pre-split table; vocab/merges come from the GGUF metadata.
    """

    var name: String
    var flavor: TokenizerFlavor
    var bos: Int
    var eos: Int
    var added: List[AddedToken]

    def __init__(
        out self,
        name: String,
        flavor: TokenizerFlavor,
        bos: Int,
        eos: Int,
    ):
        self.name = name
        self.flavor = flavor
        self.bos = bos
        self.eos = eos
        self.added = List[AddedToken]()

    def __copyinit__(out self, existing: Self):
        """Element-wise deep copy (List is not implicitly copyable)."""
        self.name = existing.name
        self.flavor = existing.flavor
        self.bos = existing.bos
        self.eos = existing.eos
        self.added = List[AddedToken]()
        for a in existing.added:
            self.added.append(a)

    def add_token(mut self, id: Int, var content: String, special: Bool):
        self.added.append(AddedToken(id, content^, special))


struct TokenizerRegistry(Movable):
    var entries: Dict[String, TokenizerSpec]

    def __init__(out self):
        self.entries = Dict[String, TokenizerSpec]()

    def register(mut self, var spec: TokenizerSpec):
        self.entries[spec.name] = spec^

    def contains(self, name: String) -> Bool:
        for key in self.entries.keys():
            if key == name:
                return True
        return False

    def apply(mut self, name: String, mut engine: BpeTokenizer) -> Bool:
        """Apply the registered spec for `name` onto `engine` in place.

        Returns False when `name` is not registered.  (In-place application
        avoids copying the spec's added-token list out of the Dict.)
        """
        if not self.contains(name):
            return False
        engine._flavor = flavor_tag(self.entries[name].flavor)
        engine._bos_id = self.entries[name].bos
        engine._eos_id = self.entries[name].eos
        engine._added = List[AddedToken]()
        for i in range(len(self.entries[name].added)):
            engine._added.append(self.entries[name].added[i])
        return True

    def size(self) -> Int:
        return len(self.entries)


def register_tokenizer(
    mut registry: TokenizerRegistry,
    var spec: TokenizerSpec,
):
    """Add a custom tokenizer to `registry` (see the module docstring)."""
    registry.register(spec^)


def make_tokenizer(
    ctx: GGUFContext,
    tokenizer_json_path: String = String(""),
    extra: Optional[TokenizerRegistry] = None,
) raises -> BpeTokenizer:
    """Select and build the tokenizer for a loaded GGUF.

    Precedence:
      1. user registry (matched on `tokenizer.ggml.model` /
         `general.architecture`) - custom tokenizers win;
      2. `tokenizer.ggml.pre` / `tokenizer.ggml.model` metadata - the three
         built-in flavors;
      3. plain GPT-2 BPE.
    A `tokenizer.json` next to the model (or at `tokenizer_json_path`)
    supplies vocab/merges/added tokens when present; otherwise everything is
    derived from the GGUF metadata alone.
    """
    var bos = get_meta_uint(ctx, "tokenizer.ggml.bos_token_id", 1)
    var model_tag = get_meta_str(ctx, "tokenizer.ggml.model", String(""))
    var arch = get_meta_str(ctx, "general.architecture", String(""))
    var flavor = detect_flavor(ctx, bos)

    # Build the flavor engine first (tokenizer.json preferred when present).
    var engine = _build_flavor(ctx, tokenizer_json_path, flavor)

    # 1. custom registry override (registered specs win; reads only, the
    #    Optional is not owned mutably here)
    if extra:
        var hit_name = model_tag
        var found = False
        for key in extra.value().entries.keys():
            if key == hit_name:
                found = True
        if not found and arch.byte_length() > 0:
            hit_name = arch
            for key in extra.value().entries.keys():
                if key == hit_name:
                    found = True
        if found:
            engine._flavor = flavor_tag(
                extra.value().entries[hit_name].flavor
            )
            engine._bos_id = extra.value().entries[hit_name].bos
            engine._eos_id = extra.value().entries[hit_name].eos
            engine._added = List[AddedToken]()
            for i in range(len(extra.value().entries[hit_name].added)):
                engine._added.append(
                    extra.value().entries[hit_name].added[i]
                )
    return engine^


def _build_flavor(
    ctx: GGUFContext, tokenizer_json_path: String, flavor: Int8
) raises -> BpeTokenizer:
    if tokenizer_json_path.byte_length() > 0:
        return BpeTokenizer.load_flavor(tokenizer_json_path, ctx, flavor)
    var engine = BpeTokenizer.load_from_gguf(ctx)
    engine._flavor = flavor
    if flavor == FLAVOR_QWEN or flavor == FLAVOR_HUNYUAN:
        engine.derive_added_tokens()
    return engine^
