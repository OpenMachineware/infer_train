# core/cli/it_cli.mojo
#
# M10: the `it-cli` binary - the llama.cpp `llama-cli`-compatible
# quick-verification CLI of infer_train: load a GGUF and generate,
# locally or across RPC workers (`-sm layer --rpc ...`).
#
# The entry is thin - argument parsing, the generation loops (local and
# distributed) and quantization live in core/cli_common, shared with
# it-server and it-rpc-server.
#
# Build:
#   make cli   # -> ./it-cli
#
# Usage:
#   it-cli -m model.gguf -p "prompt" -n 64 --temp 0.6 --top-p 0.95 ...
#   it-cli -m model.gguf -sm layer --rpc 127.0.0.1:50052 --rpc 127.0.0.1:50053
#   it-cli --help

from src.core.cli_common import (
    CliArgs,
    parse_args,
    load_model_heap,
    GenState,
    DistributedInference,
    print_banner,
    common_options_text,
    VERSION,
)
from src.core.gguf_loader import load_gguf
from src.core.transformer import load_config
from src.core.tensor import Tensor, tensor_zeros
from src.core.simd_utils import (
    autotune_width_f16,
    get_optimal_simd_width,
    is_power_of_two,
)
from src.core.jit.jit_cache import JitCache
from std.utils.static_tuple import StaticTuple
from std.sys import argv


def help_text() -> String:
    return """
it-cli - an infer_train quick-verification CLI (llama-cli-compatible)

Usage:
  it-cli -m MODEL [options]                 generate text
  it-cli -m MODEL -sm layer --rpc H:P ...   generate across RPC workers
  it-cli profile [-m MODEL] --simd-autotune SIMD width search (M8)
  it-cli profile [-m MODEL] --jit-stats     JIT compile/hit stats (M8)
  it-cli --help                             this help
  it-cli --version                          show version and exit

llama-cli-compatible options:
  -m, --model PATH          model (GGUF or .mmdl checkpoint)
  -p, --prompt TEXT         prompt (default: empty)
  -n, --predict N           tokens to generate (default 64)
  -sm, --split-mode M       layer | row   (row: not implemented yet)
      --rpc HOST:PORT       add an RPC worker endpoint (repeatable;
                             requires -sm layer; run one
                             it-rpc-server -m MODEL --port PORT per endpoint)
""" + common_options_text() + """
Serving lives in `it-server` (llama-server-compatible HTTP service);
weight requantization in `it-server quantize`.
"""


def run_local(args: CliArgs) raises:
    var mp = load_model_heap(args.model, args.ctx_size)
    print_banner(
        args.model,
        mp[0].transformer.config,
        args.ctx_size,
        mp[0].tokenizer,
    )
    if args.mode != "off":
        print("  infer-train mode:", args.mode, "lr:", args.lr)
        print(
            "  (fine-tuning runs through the finetune API; see it-server /v1/finetune)"
        )
    var g = GenState(
        mp,
        args.n_predict,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repeat_penalty,
        args.seed,
    )
    g.prefill(args.prompt)
    while not g.done:
        var t = g.next_token()
        if t < 0:
            break
        print(g.decode_token(t), end="")
    print()


def _fill_f16(mut t: Tensor[DType.float16, 2], seed: Int, scale: Float32):
    for i in range(t.numel()):
        t.set(
            i,
            Scalar[DType.float16](
                Float32((i * 7 + seed * 13) % 101) / Float32(101.0) * scale
            ),
        )


def run_profile(args: CliArgs) raises:
    """M8: CPU performance analysis.

    --simd-autotune: micro-benchmark the 64/128/256-bit SIMD widths for
    the model's row lengths (hidden / ffn) and print the fastest width.
    --jit-stats: run the fused matmul+rmsnorm path for the model FFN
    shapes and print the JIT compile count and cache hit rate (it implies
    the specialized path, since the stats measure it).
    --jit-specialize: run the benchmark through the JIT shape-specialized
    fused kernel; off by default (the generic kernel).
    """
    if not args.simd_autotune and not args.jit_stats:
        print(
            "profile: nothing to do (use --simd-autotune and/or --jit-stats)"
        )
        return
    # model row lengths when a model is given, defaults otherwise
    var hidden = 1536
    var ffn = 8960
    if args.model.byte_length() > 0:
        var ctx = load_gguf(args.model)
        var config = load_config(ctx)
        hidden = config.hidden
        ffn = config.ffn
        print(
            "profile:", args.model, "hidden:", hidden, "ffn:", ffn
        )
    # the k-loop SIMD width of the JIT benchmark: the autotuner's choice
    # for the hidden dim when --simd-autotune is also given (task 1 <->
    # task 2), the 128-bit default otherwise
    var f16_width = 128
    if args.simd_autotune:
        print("== SIMD width autotune (f16) ==")
        var dims = List[Int]()
        dims.append(hidden)
        dims.append(ffn)
        for di in range(len(dims)):
            var dim = dims[di]
            var r = autotune_width_f16(dim)
            if dim == hidden:
                f16_width = r.best
            # r.sink is printed so the compiler keeps the timed loops alive
            print(
                "  dim", dim,
                ": 64-bit", r.ns64, "ns | 128-bit", r.ns128,
                "ns | 256-bit", r.ns256, "ns -> best", r.best, "bit",
                "(heuristic:", get_optimal_simd_width(
                    dim, is_power_of_two(dim)
                ), "bit, sink", r.sink, ")",
            )
    if args.jit_stats:
        print("== JIT shape specialization stats (fused matmul+rmsnorm) ==")
        print(
            "  specialize:", args.jit_specialize,
            "k-loop width:", f16_width, "bit",
        )
        var cache = JitCache()
        var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
        var w = tensor_zeros[DType.float16, 2](
            StaticTuple[Int, 2](ffn, hidden)
        )
        _fill_f16(x, 3, Float32(1.0))
        _fill_f16(w, 4, Float32(0.02))
        # model shape: with --jit-specialize the first run compiles and the
        # rest hit the cache; without it the generic kernel runs (default)
        for i in range(3):
            _ = cache.run_fused_jit(
                x, w, Float32(1e-5), args.jit_specialize, f16_width
            )
        # an off-table shape: miss -> generic fallback, recorded
        var x2 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](2, 64))
        var w2 = tensor_zeros[DType.float16, 2](
            StaticTuple[Int, 2](128, 64)
        )
        _fill_f16(x2, 5, Float32(1.0))
        _fill_f16(w2, 6, Float32(0.02))
        _ = cache.run_fused_jit(
            x2, w2, Float32(1e-5), args.jit_specialize, f16_width
        )
        print(
            "  compiles:", cache.compiles, "hits:", cache.hits,
            "misses:", cache.misses, "hit rate:", cache.hit_rate(),
        )


def run_distributed(args: CliArgs) raises:
    """Layer-split generation over RPC workers (llama.cpp -sm layer)."""
    var d = DistributedInference(
        args.model,
        args.ctx_size,
        args.n_predict,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repeat_penalty,
        args.seed,
        args.rpc_endpoints,
    )
    try:
        d.prefill(args.prompt)
    except:
        print("error: KV cache too small for prompt + generation")
        d.close()
        return
    while not d.done:
        var t = d.next_token()
        if t < 0:
            break
        print(d.decode_token(t), end="")
    print()
    d.close()


def main() raises:
    var args = parse_args(argv(), "generate")
    if args.action == "version":
        print("it-cli " + VERSION)
        return
    if args.action == "help":
        print(help_text())
        return
    if args.action == "profile":
        run_profile(args)
        return
    if args.action == "serve":
        print(
            "hint: serving is `it-server` (llama-server-compatible HTTP service);",
        )
        print("      e.g. it-server -m MODEL --host 127.0.0.1 --port 8080")
        return
    if args.action == "quantize":
        print(
            "hint: quantization is `it-server quantize` (the renamed CLI keeps it);",
        )
        print(
            "      e.g. it-server quantize -i IN -o OUT -f Q4_K_M"
        )
        return
    if args.model.byte_length() == 0:
        print("error: -m/--model is required (try --help)")
        return
    if args.threads > 0:
        print("threads:", args.threads)
    # M8: multi-process / multi-machine RPC (llama.cpp-style -sm/--rpc)
    if args.split_mode == "row":
        print(
            "error: -sm row (row-parallel) is not implemented yet; "
            + "use -sm layer"
        )
        return
    if len(args.rpc_endpoints) > 0:
        if args.split_mode != "layer":
            print("error: --rpc requires -sm layer")
            return
        run_distributed(args)
        return
    run_local(args)
