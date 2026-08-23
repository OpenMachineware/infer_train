# runtime/inference.mojo
#
# End-to-end inference entry: load a GGUF + tokenizer.json, dequantize the
# weights, build the transformer, and run the autoregressive generation loop
# with the real Qwen2 BPE tokenizer and the M1 sampler.

from ..core.gguf_loader import load_gguf
from ..core.json import (
    parse_json_flat_file,
    flat_get_int,
    flat_get_float,
    flat_get_bool,
)
from ..core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
    build_graph,
    DEFAULT_KV_CACHE_LEN,
    ARCH_QWEN2,
)
from ..core.tokenizers import make_tokenizer
from ..core.tokenizers.bpe_engine import BpeTokenizer
from ..core.sampler import Sampler, sample_dynamic, seed_sampler
from ..core.tensor import tensor_zeros
from ..core.graph import Graph
from ..core.ops.base.op_registry import OpRegistry
from std.utils.static_tuple import StaticTuple
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.collections import Span


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


def _dir_of(path: String) -> String:
    """Directory portion of a path ('' when there is no separator)."""
    var bytes = path.as_bytes()
    var last = -1
    for i in range(len(bytes)):
        if bytes[i] == UInt8(47):  # '/'
            last = i
    if last < 0:
        return String("")
    var buf = unsafe_alloc[UInt8](last)
    for i in range(last):
        buf.unsafe_store(i, bytes[i])
    var span = Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=buf, length=last
    )
    return String(unsafe_from_utf8=span)


def load_model(path: String) raises -> Model:
    """Load a GGUF, dequantize weights, build the transformer + tokenizer."""
    var ctx = load_gguf(path)
    var config = load_config(ctx)

    # Cross-check against config.json (GGUF metadata wins when present;
    # the JSON fills in anything the GGUF header did not carry).
    var dir = _dir_of(path)
    if dir.byte_length() > 0:
        try:
            var flat = parse_json_flat_file(dir + "/config.json")
            if config.n_layers == 0:
                config.n_layers = flat_get_int(flat, "num_hidden_layers", 0)
            if config.hidden == 0:
                config.hidden = flat_get_int(flat, "hidden_size", 0)
            if config.ffn == 0:
                config.ffn = flat_get_int(flat, "intermediate_size", 0)
            if config.n_heads == 0:
                config.n_heads = flat_get_int(flat, "num_attention_heads", 0)
            if config.n_kv_heads == 0:
                config.n_kv_heads = flat_get_int(
                    flat, "num_key_value_heads", 0
                )
            if config.vocab == 0:
                config.vocab = flat_get_int(flat, "vocab_size", 0)
            if config.rope_theta == Float32(10000.0):
                config.rope_theta = Float32(
                    flat_get_float(flat, "rope_theta", 10000.0)
                )
            if config.norm_eps == Float32(1e-6):
                config.norm_eps = Float32(
                    flat_get_float(flat, "rms_norm_eps", 1e-6)
                )
        except:
            pass
    if config.n_heads > 0:
        config.head_dim = config.hidden // config.n_heads

    var weights = collect_weights(ctx)
    var model = TransformerModel(config, ctx^, DEFAULT_KV_CACHE_LEN)
    model.weights = weights^

    # M7: auto-select the tokenizer flavor from the GGUF metadata
    # (tokenizer.ggml.model / tokenizer.ggml.pre).  A tokenizer.json is only
    # consulted for the qwen2-family path (the one this repo ships a json
    # for); hunyuan-dense / qwen35 always build from the GGUF vocabulary.
    var tokenizer_json = String("")
    if config.arch == ARCH_QWEN2:
        if dir.byte_length() > 0:
            tokenizer_json = dir + "/tokenizer.json"
        else:
            tokenizer_json = "tokenizer.json"
    var tokenizer = make_tokenizer(model.ctx, tokenizer_json)

    var graph = build_graph(model)
    var registry = OpRegistry()
    registry.register_default_ops()
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
