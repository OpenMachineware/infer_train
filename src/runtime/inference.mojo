# runtime/inference.mojo
#
# End-to-end inference entry: load a GGUF, dequantize the weights, build the
# transformer, and run the autoregressive generation loop with the real Qwen2
# BPE tokenizer and the M1 sampler.
#
# The GGUF is self-contained: architecture dims, the vocabulary, the BPE
# merges and the bos/eos ids all come from its metadata
# (`qwen2.*` / `tokenizer.ggml.*`), so the engine needs no sidecar
# config.json / tokenizer.json next to the model file.

from ..core.gguf_loader import load_gguf
from ..core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
    build_graph,
    DEFAULT_KV_CACHE_LEN,
)
from ..core.tokenizers import make_tokenizer
from ..core.tokenizers.bpe_engine import BpeTokenizer
from ..core.sampler import Sampler, sample_dynamic, seed_sampler
from ..core.tensor import tensor_zeros
from ..core.graph import Graph
from ..core.ops.base.op_registry import OpRegistry
from std.utils.static_tuple import StaticTuple


struct Model(Movable):
    var transformer: TransformerModel
    var tokenizer: BpeTokenizer
    var registry: OpRegistry
    var graph: Graph

    def __init__(
        out self,
        var transformer: TransformerModel,
        var tokenizer: BpeTokenizer,
        var registry: OpRegistry,
        var graph: Graph,
    ):
        self.transformer = transformer^
        self.tokenizer = tokenizer^
        self.registry = registry^
        self.graph = graph^


def load_model(
    path: String, ctx_len: Int = DEFAULT_KV_CACHE_LEN
) raises -> Model:
    """Load a GGUF, dequantize weights, build the transformer + tokenizer.

    Everything the engine needs (architecture dims, vocab, BPE merges,
    bos/eos ids) comes from the GGUF metadata; no sidecar config.json /
    tokenizer.json is read.  `ctx_len` sizes the KV cache (the CLI's
    -c/--ctx-size; the C-API keeps the default).
    """
    var ctx = load_gguf(path)
    # head_dim comes from load_config: the metadata `attention.key_length`
    # wins (Qwen3-0.6B: 128 != hidden/n_heads = 64; qwen35: 256 != 213),
    # with hidden/n_heads as the fallback.
    var config = load_config(ctx)

    var weights = collect_weights(ctx)
    var model = TransformerModel(config, ctx^, ctx_len)
    model.weights = weights^

    # M7: auto-select the tokenizer flavor from the GGUF metadata
    # (tokenizer.ggml.model / tokenizer.ggml.pre); the vocab and merges come
    # from the GGUF arrays (tokenizer.ggml.tokens / tokenizer.ggml.merges).
    var tokenizer = make_tokenizer(model.ctx, String(""))

    var graph = build_graph(model)
    # NOTE: the registry is left unregistered on purpose.  Referencing the
    # GPU dispatch functions (register_default_ops) makes the in-process
    # Metal compiler fail ("Metal Compiler failed to compile metallib") in
    # *executable* builds - a Mojo 1.0 toolchain bug (shared-library builds
    # are unaffected).  The typed-forward path (transformer.forward, used
    # by it-cli / it-server / the C-API generate) never touches the
    # registry; every Interpreter consumer (training, the M2-M5 tests)
    # builds its own OpRegistry and calls register_default_ops itself.
    var registry = OpRegistry()
    return Model(model^, tokenizer^, registry^, graph^)


def generate(
    mut model: Model,
    prompt: String,
    max_tokens: Int,
    temperature: Float32 = Float32(0.6),
    top_p: Float32 = Float32(0.95),
    top_k: Int = 40,
    verbose: Bool = False,
    seed: Optional[Int] = None,
) raises -> String:
    """Run the autoregressive loop and return the decoded completion.

    `prompt` is encoded with a leading BOS token (Qwen2 add_bos_token);
    generation stops at the model's EOS token or after `max_tokens`.
    """
    var tokens = model.tokenizer.encode_with_bos(prompt)
    seed_sampler(seed)
    var sampler = Sampler(
        temperature=temperature, top_k=top_k, top_p=top_p
    )
    var capacity = model.transformer.cache.capacity()
    if len(tokens) + max_tokens > capacity:
        raise Error(
            "generate: KV cache capacity "
            + String(capacity)
            + " too small for "
            + String(len(tokens) + max_tokens)
            + " tokens"
        )

    var generated = List[Int]()
    var logits = tensor_zeros[DType.float32, 1](
        StaticTuple[Int, 1](model.transformer.config.vocab)
    )

    # prefill: feed the prompt one token at a time (positions 0..n-1)
    for i in range(len(tokens)):
        logits = model.transformer.forward(tokens[i], i)

    # decode loop
    for _ in range(max_tokens):
        var next_token = sample_dynamic[DType.float32](
            logits, sampler, tokens
        )
        if next_token == model.tokenizer.eos_id():
            if verbose:
                print("[eos]")
            break
        generated.append(next_token)
        tokens.append(next_token)
        if verbose:
            var one = List[Int]()
            one.append(next_token)
            print(model.tokenizer.decode(one), end="")
        logits = model.transformer.forward(next_token, len(tokens) - 1)
    if verbose:
        print()
    return model.tokenizer.decode(generated)
