# core/cli/infer_train_cli.mojo
#
# M7: the `infer_train` command line (llama.cpp-compatible core flags plus
# the private --infer-train-* group).
#
# Build:
#   pixi run mojo build -I . src/core/cli/infer_train_cli.mojo \
#       -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib -o infer_train
#
# Usage:
#   infer_train -m model.gguf -p "prompt" -n 64 --temp 0.6 --top-p 0.95 ...
#   infer_train quantize -i model.mmdl -o model_q.gguf -f Q4_K_M
#   infer_train serve -m model.gguf --host 127.0.0.1 --port 8080
#   infer_train --help

from src.core.gguf_loader import load_gguf, find_tensor
from src.core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
    DEFAULT_KV_CACHE_LEN,
)
from src.core.tokenizers import make_tokenizer
from src.core.sampler import Sampler, sample_dynamic, seed_sampler
from src.core.tensor import tensor_zeros, Tensor
from src.core.ops.quantized.dequantize import dequantize_into
from src.core.ops.quantized.requantize import requantize, QuantizedWeights
from src.core.mmdl_storage import ByteBuf, _write_gguf, _kv_string, _kv_uint32
from std.utils.static_tuple import StaticTuple
from std.sys import argv
from std.io.file import FileHandle


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
    var action: String  # generate / quantize / serve / help

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


def main() raises:
    var args = parse_args(argv())
    if args.action == "help":
        print(help_text())
        return
    if args.action == "quantize":
        quantize_file(args)
        return
    if args.action == "serve":
        print("HTTP server: run `python -m infer_train.server` (M7 Python")
        print("wrapper) with INFERTRAIN_MODEL='" + args.model + "' and")
        print("INFERTRAIN_HOST='" + args.host + "' INFERTRAIN_PORT='" + String(args.port) + "'.")
        print("OpenAI-compatible endpoints: /v1/models /v1/chat/completions")
        print("/v1/completions /v1/finetune /v1/finetune/status.")
        return
    if args.model.byte_length() == 0:
        print("error: -m/--model is required (try --help)")
        return
    generate(args)


def help_text() -> String:
    return """
infer_train - a Mojo 1.0 inference + training engine (M7)

Usage:
  infer_train -m MODEL [options]                 generate text
  infer_train quantize -i IN -o OUT -f FORMAT    requantize weights
  infer_train serve -m MODEL [--host H] [--port P]   HTTP server note
  infer_train --help                             this help

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

infer-train options:
      --infer-train-mode M  off | lora | full   (default off)
      --infer-train-lr F    fine-tuning learning rate (default 1e-5)
      --infer-train-lora-path P   LoRA adapter checkpoint (.mmdl)
      --infer-train-accum-steps N  gradient accumulation steps

quantize subcommand:
  -i, --input PATH          source model (.mmdl / .gguf)
  -o, --output PATH         output model
  -f, --format F            Q4_K_M | Q8_0 | NF4   (default Q4_K_M)
"""


def parse_args(arg_span: Span[StringSpan[ImmStaticOrigin], ImmStaticOrigin]) raises -> CliArgs:
    var arg_list = List[String]()
    for a in arg_span:
        arg_list.append(String(a))
    var args = CliArgs()
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
        if a == "quantize":
            args.action = "quantize"
        elif a == "serve":
            args.action = "serve"
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


def generate(args: CliArgs) raises:
    from std.os import getenv

    if args.threads > 0:
        print("threads:", args.threads)
    _ = args.threads
    var ctx = load_gguf(args.model)
    var config = load_config(ctx)
    print("loaded:", args.model)
    print("  arch layers:", config.n_layers, "hidden:", config.hidden, "vocab:", config.vocab)
    var weights = collect_weights(ctx)
    var model = TransformerModel(config, ctx^, args.ctx_size)
    model.weights = weights^
    var tokenizer = make_tokenizer(model.ctx, String(""))
    print("  tokenizer:", tokenizer.flavor_name(), "ctx:", args.ctx_size)
    if args.mode != "off":
        print("  infer-train mode:", args.mode, "lr:", args.lr)
        print("  (fine-tuning runs through the finetune API; see /v1/finetune)")

    var tokens = tokenizer.encode_with_bos(args.prompt)
    seed_sampler(Optional(args.seed) if args.seed >= 0 else None)
    var sampler = Sampler(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
    )
    var vocab = config.vocab
    var logits = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](vocab))
    for i in range(len(tokens)):
        logits = model.forward(tokens[i], i)
    var generated = List[Int]()
    var counts = Dict[Int, Int]()
    for step in range(args.n_predict):
        # repeat penalty over the generated history
        if args.repeat_penalty != Float32(1.0):
            var adjusted = tensor_zeros[DType.float32, 1](
                StaticTuple[Int, 1](vocab)
            )
            for i in range(vocab):
                var v = Float32(logits.get(i))
                var c = counts.get(i, 0)
                if c > 0:
                    if v > Float32(0):
                        v = v / args.repeat_penalty
                    else:
                        v = v * args.repeat_penalty
                adjusted.set(i, Scalar[DType.float32](v))
            logits = adjusted
        var next_token = sample_dynamic[DType.float32](logits, sampler, tokens)
        if next_token == tokenizer.eos_id():
            break
        generated.append(next_token)
        tokens.append(next_token)
        counts[next_token] = counts.get(next_token, 0) + 1
        var one = List[Int]()
        one.append(next_token)
        print(tokenizer.decode(one), end="")
        logits = model.forward(next_token, len(tokens) - 1)
    print()


def quantize_file(args: CliArgs) raises:
    """Requantize every F16/F32 weight tensor of a GGUF/.mmdl model."""
    if args.input.byte_length() == 0 or args.output.byte_length() == 0:
        print("error: quantize needs -i INPUT and -o OUTPUT")
        return
    print("quantize:", args.input, "->", args.output, "format:", args.quant_format)
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
            dequantize_into(
                t.ggml_type, ctx.data, ctx.data_offset + t.offset, fp16, numel
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
            dequantize_into(
                t.ggml_type, ctx.data, ctx.data_offset + t.offset, fp16, numel
            )
            types.append(1)
            for i in range(numel):
                var u = _f16_bits(Float32(fp16.get(i)))
                b.append_u8(UInt8(u & 0xFF))
                b.append_u8(UInt8((u >> 8) & 0xFF))
        else:
            # keep already-quantized tensors as-is (bulk copy)
            types.append(t.ggml_type)
            var raw = ctx.data.unsafe_offset(ctx.data_offset + t.offset)
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
