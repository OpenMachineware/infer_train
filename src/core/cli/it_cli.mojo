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
from std.sys import argv


def help_text() -> String:
    return """
it-cli - an infer_train quick-verification CLI (llama-cli-compatible)

Usage:
  it-cli -m MODEL [options]                 generate text
  it-cli -m MODEL -sm layer --rpc H:P ...   generate across RPC workers
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
