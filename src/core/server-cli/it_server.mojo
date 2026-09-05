# core/server-cli/it_server.mojo
#
# M10: the `it-server` binary - the llama.cpp `llama-server`-compatible
# HTTP service of infer_train (OpenAI-compatible endpoints, native Mojo
# implementation; see core/http for the server and core/cli_common for
# the shared CLI/generation code it-cli and it-rpc-server also use).
#
# Build:
#   make server        # -> ./it-server
#   make infer_train   # -> ./infer_train (legacy alias, same entry)
#
# Usage:
#   it-server -m model.gguf --host 127.0.0.1 --port 8080
#   it-server -m model.gguf --api-key sk-... --alias my-model
#   it-server quantize -i model.mmdl -o model_q.gguf -f Q4_K_M
#   it-server --help
#
# The multi-process / multi-machine RPC worker is a separate binary:
# it-rpc-server (see core/server-cli/it_rpc_server.mojo).

from src.core.cli_common import (
    CliArgs,
    parse_args,
    load_model_heap,
    quantize_file,
    basename,
    common_options_text,
    VERSION,
)
from src.core.http import ServerState
from std.sys import argv
from std.os import getenv


def help_text() -> String:
    return (
        """
it-server - an infer_train HTTP server (llama-server-compatible)

Usage:
  it-server -m MODEL [options]                start the HTTP service
  it-server quantize -i IN -o OUT -f FORMAT   requantize weights
  it-server --help                            this help
  it-server --version                         show version and exit

llama-server-compatible options:
      --host H            bind address (default 127.0.0.1)
      --port N            TCP port (default 8080)
      --alias NAME        model name advertised by /v1/models
                          (default: the model file's basename)
      --api-key KEY       accepted API key (repeatable; also the
                          INFERTRAIN_API_KEY env var)
      --no-webui          accepted (the Mojo server has no WebUI)
      --metrics           accepted (ignored)
"""
        + common_options_text()
        + """
Endpoints (OpenAI-compatible):
  GET  /health
  GET  /v1/models
  POST /v1/completions           (+ "stream": true for SSE)
  POST /v1/chat/completions
  POST /v1/finetune              (synchronous; recorded in /v1/finetune/status)
  GET  /v1/finetune/status

quantize subcommand:
  -i, --input PATH          source model (.mmdl / .gguf)
  -o, --output PATH         output model
  -f, --format F            Q4_K_M | Q8_0 | NF4   (default Q4_K_M)

Local generation lives in `it-cli` (llama-cli-compatible); the
multi-process / multi-machine RPC worker is `it-rpc-server`.
"""
    )


def main() raises:
    var args = parse_args(argv(), "serve")
    if args.action == "version":
        print("it-server " + VERSION)
        return
    if args.action == "help":
        print(help_text())
        return
    if args.action == "quantize":
        quantize_file(args)
        return
    if args.action == "generate":
        print(
            "hint: local generation is `it-cli` (llama-cli-compatible);",
        )
        print('      e.g. it-cli -m MODEL -p "prompt" -n 64')
        return
    if args.model.byte_length() == 0:
        print("error: -m/--model is required (try --help)")
        return

    # API keys: --api-key flags, else the INFERTRAIN_API_KEY env var.
    # (Copy out of args - the parser struct is still used below;
    # getenv returns "" when unset.)
    var api_keys = args.api_keys.copy()
    if len(api_keys) == 0:
        var env_key = getenv("INFERTRAIN_API_KEY")
        if env_key.byte_length() > 0:
            api_keys.append(env_key)
    var name = args.model_alias
    if name.byte_length() == 0:
        name = basename(args.model)

    var model_p = load_model_heap(args.model, args.ctx_size)
    print("loaded:", args.model)
    print(
        "  arch layers:",
        model_p[unsafe_offset=0].transformer.config.n_layers,
        "hidden:",
        model_p[unsafe_offset=0].transformer.config.hidden,
        "vocab:",
        model_p[unsafe_offset=0].transformer.config.vocab,
    )
    var server = ServerState(
        model_p,
        name,
        args.model,
        api_keys,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repeat_penalty,
    )
    server.run(args.host, args.port)
