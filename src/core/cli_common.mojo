# core/cli_common.mojo
#
# Shared CLI infrastructure for the three infer_train entry points:
#
#   * it-server     (src/core/server-cli/it_server.mojo)
#                   llama-server-compatible HTTP service (see core/http)
#   * it-cli        (src/core/cli/it_cli.mojo)
#                   llama-cli-compatible quick-verification generation
#   * it-rpc-server (src/core/server-cli/it_rpc_server.mojo)
#                   the RPC worker (llama.cpp `llama-rpc-server`-compatible)
#
# The entries are deliberately thin: argument parsing, the local and the
# distributed (layer-split RPC) generation loops, and the quantize tool all
# live here, so the three binaries share one implementation (different
# entry points, not three copies of the code).
#
# Mojo 1.0 notes honored here (same conventions as the rest of the tree):
# `def`-only, no runtime globals, `String("...")` for runtime strings,
# explicit moves (`^`) for Movable values.

from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    TransformerConfig,
    load_config,
    collect_weights,
    DEFAULT_KV_CACHE_LEN,
)
from src.core.rpc import RpcClient
from src.core.tokenizers import make_tokenizer
from src.core.tokenizers.bpe_engine import BpeTokenizer
from src.core.sampler import Sampler, sample_dynamic, seed_sampler
from src.core.tensor import tensor_zeros, Tensor
from src.core.ops.quantized.dequantize import dequantize_into
from src.core.ops.quantized.requantize import requantize, QuantizedWeights
from src.core.mmdl_storage import ByteBuf, _write_gguf
from src.runtime.inference import Model, load_model
from src.version import VERSION
from std.utils.static_tuple import StaticTuple
from std.sys import argv
from std.io.file import FileHandle
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.collections import Span


# -- command-line arguments ----------------------------------------------------


struct CliArgs(Movable):
    var model: String
    var prompt: String
    var n_predict: Int
    var ctx_size: Int
    var temperature: Float32
    var top_p: Float32
    var top_k: Int
    var repeat_penalty: Float32
    var threads: Int
    var seed: Int
    var host: String
    var port: Int
    var no_cnv: Bool
    var mode: String  # infer-train mode: off/lora/full
    var lora_path: String
    var lr: Float32
    var accum_steps: Int
    var quant_format: String
    var input: String
    var output: String
    var action: String  # generate / quantize / serve / help / version
    # M8: multi-process / multi-machine RPC (llama.cpp-style)
    var split_mode: String  # none / layer / row
    var rpc_endpoints: List[String]  # --rpc HOST:PORT (repeatable)
    # llama-server-compatible additions (it-server)
    var model_alias: String  # --alias: model name advertised by /v1/models
    var api_keys: List[String]  # --api-key (repeatable)
    var no_webui: Bool  # accepted (the Mojo server has no WebUI)
    var metrics: Bool  # accepted (ignored)

    def __init__(out self):
        self.model = String("")
        self.prompt = String("")
        self.n_predict = 64
        self.ctx_size = 512
        self.temperature = Float32(0.6)
        self.top_p = Float32(0.95)
        self.top_k = 40
        self.repeat_penalty = Float32(1.0)
        self.threads = 0
        self.seed = -1
        self.host = String("127.0.0.1")
        self.port = 8080
        self.no_cnv = False
        self.mode = String("off")
        self.lora_path = String("")
        self.lr = Float32(1e-5)
        self.accum_steps = 1
        self.quant_format = String("Q4_K_M")
        self.input = String("")
        self.output = String("")
        self.action = String("generate")
        self.split_mode = String("none")
        self.rpc_endpoints = List[String]()
        self.model_alias = String("")
        self.api_keys = List[String]()
        self.no_webui = False
        self.metrics = False


def parse_args(
    arg_span: Span[StringSpan[ImmStaticOrigin], ImmStaticOrigin],
    default_action: String,
) raises -> CliArgs:
    """Parse argv for any of the three entries (one parser, shared flags).

    `default_action` is the entry's role: "generate" for it-cli, "serve"
    for it-server.  Subcommands (quantize / serve) override it.
    """
    var arg_list = List[String]()
    for a in arg_span:
        arg_list.append(String(a))
    var args = CliArgs()
    args.action = default_action
    var i = 0
    var positional = 0
    while i < len(arg_list):
        var a = arg_list[i]
        if positional == 0 and i == 0:
            positional = 1
            i += 1
            continue  # argv[0] = program name
        if a == "--help" or a == "-h":
            args.action = "help"
            return args^
        if a == "--version" or a == "-V":
            args.action = "version"
            return args^
        if a == "quantize":
            args.action = "quantize"
        elif a == "serve":
            args.action = "serve"
        elif a == "generate":
            args.action = "generate"  # it-server: hint to it-cli; it-cli: no-op
        elif a == "-m" or a == "--model":
            args.model = _next(arg_list, i)
            i += 1
        elif a == "-p" or a == "--prompt":
            args.prompt = _next(arg_list, i)
            i += 1
        elif a == "-c" or a == "--ctx-size":
            args.ctx_size = _next_int(arg_list, i)
            i += 1
        elif a == "-n" or a == "--predict":
            args.n_predict = _next_int(arg_list, i)
            i += 1
        elif a == "--temp":
            args.temperature = _next_f32(arg_list, i)
            i += 1
        elif a == "--top-p":
            args.top_p = _next_f32(arg_list, i)
            i += 1
        elif a == "--top-k":
            args.top_k = _next_int(arg_list, i)
            i += 1
        elif a == "--repeat-penalty":
            args.repeat_penalty = _next_f32(arg_list, i)
            i += 1
        elif a == "-t" or a == "--threads":
            args.threads = _next_int(arg_list, i)
            i += 1
        elif a == "--seed":
            args.seed = _next_int(arg_list, i)
            i += 1
        elif a == "--host":
            args.host = _next(arg_list, i)
            i += 1
        elif a == "--port":
            args.port = _next_int(arg_list, i)
            i += 1
        elif a == "--no-cnv":
            args.no_cnv = True
        elif a == "-np" or a == "--parallel":
            _ = _next_int(arg_list, i)
            i += 1
        elif a == "-sm" or a == "--split-mode":
            args.split_mode = _next(arg_list, i)
            i += 1
        elif a == "--rpc":
            args.rpc_endpoints.append(_next(arg_list, i))
            i += 1
        elif a == "--alias":
            args.model_alias = _next(arg_list, i)
            i += 1
        elif a == "--api-key":
            args.api_keys.append(_next(arg_list, i))
            i += 1
        elif a == "--no-webui":
            args.no_webui = True
        elif a == "--metrics":
            args.metrics = True
        elif a == "--infer-train-mode":
            args.mode = _next(arg_list, i)
            i += 1
        elif a == "--infer-train-lr":
            args.lr = _next_f32(arg_list, i)
            i += 1
        elif a == "--infer-train-lora-path":
            args.lora_path = _next(arg_list, i)
            i += 1
        elif a == "--infer-train-accum-steps":
            args.accum_steps = _next_int(arg_list, i)
            i += 1
        elif a == "-i" or a == "--input":
            args.input = _next(arg_list, i)
            i += 1
        elif a == "-o" or a == "--output":
            args.output = _next(arg_list, i)
            i += 1
        elif a == "-f" or a == "--format":
            args.quant_format = _next(arg_list, i)
            i += 1
        else:
            print("warning: unknown argument:", a)
        i += 1
    return args^


def _next(arg_list: List[String], i: Int) -> String:
    if i + 1 >= len(arg_list):
        print("error: missing value after", arg_list[i])
        return String("")
    return arg_list[i + 1]


def _next_int(arg_list: List[String], i: Int) raises -> Int:
    var s = _next(arg_list, i)
    var v = 0
    var neg = False
    var s_bytes = s.as_bytes()
    for c in range(len(s_bytes)):
        var b = Int(s_bytes[c])
        if b == 45:
            neg = True
        elif b >= 48 and b <= 57:
            v = v * 10 + (b - 48)
    return -v if neg else v


def _next_f32(arg_list: List[String], i: Int) raises -> Float32:
    var s = _next(arg_list, i)
    var whole = 0
    var frac = 0
    var frac_digits = 0
    var neg = False
    var seen_dot = False
    var s_bytes = s.as_bytes()
    for c in range(len(s_bytes)):
        var b = Int(s_bytes[c])
        if b == 45:
            neg = True
        elif b == 46:
            seen_dot = True
        elif b >= 48 and b <= 57:
            if seen_dot:
                frac = frac * 10 + (b - 48)
                frac_digits += 1
            else:
                whole = whole * 10 + (b - 48)
    var f = Float32(whole)
    if frac_digits > 0:
        var divisor = Float32(1.0)
        for _ in range(frac_digits):
            divisor = divisor * Float32(10.0)
        f = f + Float32(frac) / divisor
    return -f if neg else f


# -- shared model loading -------------------------------------------------------


def load_model_heap(
    path: String, ctx_size: Int
) raises -> Pointer[Model, MutUntrackedOrigin]:
    """Load the model into one heap slot and return the owning pointer.

    Both it-cli (one run) and it-server (one load, many requests) use this:
    the generation state below holds the pointer, so the same code drives
    a single-shot CLI run and a long-lived server.  Like the C-API
    (infer_train_load_model) the slot is not freed explicitly - the engine
    buffers are MutUntrackedOrigin and the process reclaims them at exit.
    """
    var mp = unsafe_alloc[Model](1)
    mp[0] = load_model(path, ctx_size)
    return mp^


def print_banner(
    model_path: String,
    config: TransformerConfig,
    ctx_size: Int,
    tokenizer: BpeTokenizer,
):
    """The standard load banner (test_rpc.sh greps the `tokenizer:` line)."""
    print("loaded:", model_path)
    print(
        "  arch layers:",
        config.n_layers,
        "hidden:",
        config.hidden,
        "vocab:",
        config.vocab,
    )
    print("  tokenizer:", tokenizer.flavor_name(), "ctx:", ctx_size)


# -- generation state (local) ---------------------------------------------------
#
# One GenState is one in-flight generation: the KV cache state (inside the
# shared model) plus the token history, the repeat-penalty counts and the
# sampler.  `prefill` feeds the prompt, `next_token` produces one token
# (-1 = EOS / limit reached), so each entry point owns its output handling:
# it-cli prints per token, it-server collects or streams SSE frames.


def adjust_repeat_penalty(
    mut logits: Tensor[DType.float32, 1],
    counts: Dict[Int, Int],
    penalty: Float32,
    vocab: Int,
) raises:
    """Divide/multiply the logits of already-generated tokens (llama.cpp).

    Takes the tensor by mutable reference and re-assigns it in place
    (a fresh tensor when the penalty is active), so the caller's
    `self.logits` / local `logits` is updated without a move-out.
    """
    if penalty == Float32(1.0):
        return
    var adjusted = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](vocab))
    for i in range(vocab):
        var v = Float32(logits.get(i))
        var c = counts.get(i, 0)
        if c > 0:
            if v > Float32(0):
                v = v / penalty
            else:
                v = v * penalty
        adjusted.set(i, Scalar[DType.float32](v))
    logits = adjusted^


struct GenState(Movable):
    var model_p: Pointer[Model, MutUntrackedOrigin]
    var sampler: Sampler
    var tokens: List[Int]
    var counts: Dict[Int, Int]
    var generated: List[Int]
    var logits: Tensor[DType.float32, 1]
    var repeat_penalty: Float32
    var n_predict: Int
    var step_count: Int
    var done: Bool

    def __init__(
        out self,
        model_p: Pointer[Model, MutUntrackedOrigin],
        n_predict: Int,
        temperature: Float32,
        top_p: Float32,
        top_k: Int,
        repeat_penalty: Float32,
        seed: Int,
    ) raises:
        var vocab = model_p[0].transformer.config.vocab
        seed_sampler(Optional(seed) if seed >= 0 else None)
        self.model_p = model_p
        self.sampler = Sampler(
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        self.tokens = List[Int]()
        self.counts = Dict[Int, Int]()
        self.generated = List[Int]()
        self.logits = tensor_zeros[DType.float32, 1](
            StaticTuple[Int, 1](vocab)
        )
        self.repeat_penalty = repeat_penalty
        self.n_predict = n_predict
        self.step_count = 0
        self.done = False

    def prefill(mut self, prompt: String) raises:
        var tokens = self.model_p[0].tokenizer.encode_with_bos(prompt)
        var capacity = self.model_p[0].transformer.cache.capacity()
        if len(tokens) + self.n_predict > capacity:
            raise Error(
                "KV cache capacity " + String(capacity)
                + " too small for "
                + String(len(tokens) + self.n_predict)
                + " tokens"
            )
        self.tokens = tokens^
        for i in range(len(self.tokens)):
            self.logits = self.model_p[0].transformer.forward(
                self.tokens[i], i
            )

    def next_token(mut self) raises -> Int:
        """One decode step; returns the token, or -1 at EOS / the limit."""
        if self.done or self.step_count >= self.n_predict:
            self.done = True
            return -1
        var vocab = self.model_p[0].transformer.config.vocab
        adjust_repeat_penalty(
            self.logits, self.counts, self.repeat_penalty, vocab
        )
        var t = sample_dynamic[DType.float32](
            self.logits, self.sampler, self.tokens
        )
        if t == self.model_p[0].tokenizer.eos_id():
            self.done = True
            return -1
        self.tokens.append(t)
        self.counts[t] = self.counts.get(t, 0) + 1
        self.generated.append(t)
        self.step_count += 1
        self.logits = self.model_p[0].transformer.forward(
            t, len(self.tokens) - 1
        )
        return t

    def decode_token(self, t: Int) -> String:
        var one = List[Int]()
        one.append(t)
        return self.model_p[0].tokenizer.decode(one)

    def decode_all(self) -> String:
        return self.model_p[0].tokenizer.decode(self.generated)


# -- generation state (distributed, -sm layer --rpc) ----------------------------
#
# The master keeps the embedding + output head and the generation loop;
# each `--rpc` worker owns a contiguous layer range (and its KV/SSM state)
# and is chained per token.  The fp16 hidden state crosses the wire
# losslessly, so the output is numerically identical to the local run.


struct DistributedInference(Movable):
    var master: TransformerModel
    var config: TransformerConfig
    var tokenizer: BpeTokenizer
    var workers: List[RpcClient]
    var sampler: Sampler
    var tokens: List[Int]
    var counts: Dict[Int, Int]
    var generated: List[Int]
    var x: Tensor[DType.float16, 2]
    var ctx_size: Int
    var repeat_penalty: Float32
    var n_predict: Int
    var step_count: Int
    var done: Bool

    def __init__(
        out self,
        model_path: String,
        ctx_size: Int,
        n_predict: Int,
        temperature: Float32,
        top_p: Float32,
        top_k: Int,
        repeat_penalty: Float32,
        seed: Int,
        endpoints: List[String],
    ) raises:
        var ctx = load_gguf(model_path)
        var config = load_config(ctx)
        print("loaded:", model_path)
        print(
            "  arch layers:",
            config.n_layers,
            "hidden:",
            config.hidden,
            "vocab:",
            config.vocab,
        )
        var n = len(endpoints)
        var base = config.n_layers // n
        var rem = config.n_layers % n
        # Connect, assign contiguous layer ranges (the first `rem` workers
        # get one extra layer), and make each worker load its shard.
        var workers = List[RpcClient]()
        for i in range(n):
            var (host, port) = split_host_port(endpoints[i])
            var client = RpcClient()
            client.connect(host, port)
            var lo = i * base + _min(i, rem)
            var hi = (i + 1) * base + _min(i + 1, rem)
            client.init_shard(lo, hi, ctx_size)
            print(
                "  rpc[" + String(i) + "] " + host + ":" + String(port)
                + ": layers " + String(lo) + ".." + String(hi - 1)
            )
            workers.append(client^)
        print("  split mode: layer (" + String(n) + " workers)")
        # Master model: embedding + output head only (no layers, no KV).
        var master = TransformerModel(
            config, ctx^, ctx_size, shard_lo=0, shard_hi=0, load_heads=True
        )
        var tokenizer = make_tokenizer(master.ctx, String(""))
        print("  tokenizer:", tokenizer.flavor_name(), "ctx:", ctx_size)
        seed_sampler(Optional(seed) if seed >= 0 else None)
        self.master = master^
        self.config = config
        self.tokenizer = tokenizer^
        self.workers = workers^
        self.sampler = Sampler(
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        self.tokens = List[Int]()
        self.counts = Dict[Int, Int]()
        self.generated = List[Int]()
        self.x = tensor_zeros[DType.float16, 2](
            StaticTuple[Int, 2](1, config.hidden)
        )
        self.ctx_size = ctx_size
        self.repeat_penalty = repeat_penalty
        self.n_predict = n_predict
        self.step_count = 0
        self.done = False

    def prefill(mut self, prompt: String) raises:
        self.tokens = self.tokenizer.encode_with_bos(prompt)
        if len(self.tokens) + self.n_predict > self.ctx_size:
            raise Error(
                "KV cache capacity " + String(self.ctx_size)
                + " too small for "
                + String(len(self.tokens) + self.n_predict)
                + " tokens"
            )
        # Chain the workers per prompt token (no logits needed).
        var x = self.master.embed(self.tokens[0])
        for wi in range(len(self.workers)):
            x = self.workers[wi].forward(0, x)
        for i in range(1, len(self.tokens)):
            x = self.master.embed(self.tokens[i])
            for wi in range(len(self.workers)):
                x = self.workers[wi].forward(i, x)
        self.x = x^

    def next_token(mut self) raises -> Int:
        """One decode step; returns the token, or -1 at EOS / the limit."""
        if self.done or self.step_count >= self.n_predict:
            self.done = True
            return -1
        var logits = self.master.head(self.x)
        adjust_repeat_penalty(
            logits, self.counts, self.repeat_penalty, self.config.vocab
        )
        var t = sample_dynamic[DType.float32](
            logits, self.sampler, self.tokens
        )
        if t == self.tokenizer.eos_id():
            self.done = True
            return -1
        self.tokens.append(t)
        self.counts[t] = self.counts.get(t, 0) + 1
        self.generated.append(t)
        self.step_count += 1
        self.x = self.master.embed(t)
        for wi in range(len(self.workers)):
            self.x = self.workers[wi].forward(len(self.tokens) - 1, self.x)
        return t

    def decode_token(self, t: Int) -> String:
        var one = List[Int]()
        one.append(t)
        return self.tokenizer.decode(one)

    def decode_all(self) -> String:
        return self.tokenizer.decode(self.generated)

    def close(mut self):
        for wi in range(len(self.workers)):
            self.workers[wi].close()


# -- helpers ---------------------------------------------------------------------


def split_host_port(s: String) raises -> Tuple[String, Int]:
    """Split a `host:port` RPC endpoint (port = digits after last ':')."""
    var bytes = s.as_bytes()
    var last = -1
    for i in range(len(bytes)):
        if bytes[i] == UInt8(58):  # ':'
            last = i
    if last < 0:
        raise Error("bad --rpc endpoint (want host:port): " + s)
    var host = String("")
    if last > 0:
        var buf = unsafe_alloc[UInt8](last)
        for i in range(last):
            buf.unsafe_store(i, bytes[i])
        var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=last)
        host = String(unsafe_from_utf8=span)
        buf.unsafe_free()
    var port = 0
    for i in range(last + 1, len(bytes)):
        var b = Int(bytes[i])
        if b < 48 or b > 57:
            raise Error("bad --rpc port in: " + s)
        port = port * 10 + (b - 48)
    if port <= 0 or port > 65535:
        raise Error("bad --rpc port in: " + s)
    return (host, port)


def _min(a: Int, b: Int) -> Int:
    return a if a < b else b


def basename(path: String) -> String:
    """The last `/`-separated component (model name for /v1/models)."""
    var bytes = path.as_bytes()
    var last = -1
    for i in range(len(bytes)):
        if bytes[i] == UInt8(47):  # '/'
            last = i
    if last < 0 or last == len(bytes) - 1:
        return path
    var n = len(bytes) - last - 1
    var buf = unsafe_alloc[UInt8](n)
    for i in range(n):
        buf.unsafe_store(i, bytes[last + 1 + i])
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=n)
    var out = String(unsafe_from_utf8=span)
    buf.unsafe_free()
    return out


# -- quantize (shared tool, exposed via `it-server quantize`) ---------------------


def quantize_file(args: CliArgs) raises:
    """Requantize every F16/F32 weight tensor of a GGUF/.mmdl model."""
    if args.input.byte_length() == 0 or args.output.byte_length() == 0:
        print("error: quantize needs -i INPUT and -o OUTPUT")
        return
    print(
        "quantize:",
        args.input,
        "->",
        args.output,
        "format:",
        args.quant_format,
    )
    var ctx = load_gguf(args.input)
    var kv = ByteBuf(1024)
    from src.core.mmdl_storage import _kv_from_meta

    for key in ctx.metadata.keys():
        _kv_from_meta(kv, key, ctx.metadata[key], ctx)
    var names = List[String]()
    var dims = List[List[Int]]()
    var types = List[Int]()
    var bytes = List[ByteBuf]()
    for t in ctx.tensors:
        var numel = 1
        for d in range(t.n_dims):
            numel *= t.dims[d]
        names.append(t.name)
        var dl = List[Int]()
        for d in range(t.n_dims):
            dl.append(t.dims[d])
        dims.append(dl^)
        var b = ByteBuf(64)
        var block = 256
        if args.quant_format == "Q8_0":
            block = 32
        elif args.quant_format == "NF4":
            block = 64
        if (t.ggml_type == 1 or t.ggml_type == 0) and numel >= block and numel % block == 0:
            # F16 / F32 -> requantize (block-aligned tensors only)
            var fp16 = tensor_zeros[DType.float16, 2](
                StaticTuple[Int, 2](1, numel)
            )
            var (src, off) = ctx.tensor_data(t)
            dequantize_into(
                t.ggml_type, src, off, fp16, numel
            )
            var src1 = fp16.reshape[1](StaticTuple[Int, 1](numel))
            var quant = requantize(src1, numel, args.quant_format)
            types.append(quant.ggml_type)
            for u in quant.data:
                b.append_u8(u)
        elif t.ggml_type == 0:
            # small F32 tensor: convert to F16 (keeps the file lean)
            var fp16 = tensor_zeros[DType.float16, 2](
                StaticTuple[Int, 2](1, numel)
            )
            var (src, off) = ctx.tensor_data(t)
            dequantize_into(
                t.ggml_type, src, off, fp16, numel
            )
            types.append(1)
            for i in range(numel):
                var u = _f16_bits(Float32(fp16.get(i)))
                b.append_u8(UInt8(u & 0xFF))
                b.append_u8(UInt8((u >> 8) & 0xFF))
        else:
            # keep already-quantized tensors as-is (bulk copy)
            types.append(t.ggml_type)
            var raw = ctx.tensor_data_ptr(t)
            var src_bytes = _quant_bytes(t.ggml_type, numel)
            for i in range(src_bytes):
                b.append_u8(raw.unsafe_load[width=1](offset=i))
        bytes.append(b^)
    var handle = FileHandle(args.output, "w")
    _write_gguf(handle, kv, names, dims, types, bytes)
    handle.close()
    print("wrote", args.output, "(", len(names), "tensors )")


def _f16_bits(v: Float32) -> Int:
    from src.core.ops.quantized.requantize import _f16_to_u16

    return _f16_to_u16(Scalar[DType.float16](v))


def _quant_bytes(ggml_type: Int, numel: Int) -> Int:
    if ggml_type == 0:
        return numel * 4
    if ggml_type == 1:
        return numel * 2
    if ggml_type == 12:
        return (numel // 256) * 144
    if ggml_type == 13:
        return (numel // 256) * 176
    if ggml_type == 14:
        return (numel // 256) * 210
    if ggml_type == 8:
        return (numel // 32) * 34
    if ggml_type == 20:
        return (numel // 32) * 18
    if ggml_type == 23:
        return (numel // 256) * 136
    if ggml_type == 30:
        return (numel // 64) * 34
    print("warning: unsupported type", ggml_type, "- dropping tensor")
    return 0


# -- shared help text -------------------------------------------------------------


def common_options_text() -> String:
    """The option block shared by all three entries (llama.cpp-compatible
    core flags + the private --infer-train-* group)."""
    return """
llama.cpp-compatible options:
  -m, --model PATH          model (GGUF or .mmdl checkpoint)
  -p, --prompt TEXT         prompt (default: empty)
  -c, --ctx-size N          KV context length (default 512)
  -n, --predict N           tokens to generate (default 64)
      --temp F              temperature (default 0.6)
      --top-p F             top-p (default 0.95)
      --top-k N             top-k (default 40; 0 = off)
      --repeat-penalty F    repeat penalty (default 1.0 = off)
  -t, --threads N           worker threads (default: perf cores)
      --seed N              sampling seed (default: random)
      --no-cnv              raw completion mode (no conversation)
  -np, --parallel N         parallel slots (accepted; single-slot engine)
  -sm, --split-mode M       layer | row   (row: not implemented yet)
      --rpc HOST:PORT       add an RPC worker endpoint (repeatable;
                             requires -sm layer; run one
                             it-rpc-server -m MODEL --port PORT per endpoint)

infer-train options:
      --infer-train-mode M  off | lora | full   (default off)
      --infer-train-lr F    fine-tuning learning rate (default 1e-5)
      --infer-train-lora-path P   LoRA adapter checkpoint (.mmdl)
      --infer-train-accum-steps N  gradient accumulation steps
"""
