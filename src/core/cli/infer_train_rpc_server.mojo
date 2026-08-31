# core/cli/infer_train_rpc_server.mojo
#
# M8: the RPC worker (the llama.cpp-style `llama-rpc-server`).
#
# A standalone worker process that serves one master connection over TCP.
# Like llama.cpp's rpc server it is a *device* process - but instead of a
# bare device it loads the model itself (mmap'ing the same GGUF file the
# master uses) and, at INIT, takes ownership of a contiguous layer range
# [lo, hi) plus the KV/SSM state of those layers.  The master (the
# `infer_train -sm layer --rpc ...` CLI) then chains the workers per token.
#
# Protocol (see core/rpc/net.mojo): INIT -> FORWARD* -> RESET/PING, one
# framed request/response at a time.
#
# Build:
#   pixi run mojo build -I . src/core/cli/infer_train_rpc_server.mojo \
#       -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib \
#       -o infer_train_rpc_server
#
# Usage:
#   infer_train_rpc_server -m model.gguf [--port 50052] [--host 0.0.0.0]
#   infer_train_rpc_server --help

from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    load_config,
    arch_name,
)
from src.core.rpc import (
    tcp_listen,
    tcp_accept,
    tcp_close,
    send_msg,
    recv_msg,
    append_i32,
    read_i32_le,
    f16_tensor_to_bytes,
    bytes_to_f16_tensor,
    CMD_PING,
    CMD_INIT,
    CMD_FORWARD,
    CMD_RESET,
    RPC_OK,
    RPC_ERR,
)
from src.version import VERSION
from std.sys import argv


def help_text() -> String:
    return """
infer_train_rpc_server - an RPC worker for the infer_train layer split

Usage:
  infer_train_rpc_server -m MODEL [options]

Options:
  -m, --model PATH    model file (GGUF), shared with the master
      --port N        TCP port (default 50052)
      --host H        bind address (default 0.0.0.0)
  -h, --help          this help
  -V, --version       show version and exit

The master assigns the layer range at connect time (INIT); this process
serves exactly one master connection.
"""


def main() raises:
    var arg_list = List[String]()
    for a in argv():
        arg_list.append(String(a))
    var model_path = String("")
    var port = 50052
    var host = String("0.0.0.0")
    var i = 1
    while i < len(arg_list):
        var a = arg_list[i]
        if a == "--help" or a == "-h":
            print(help_text())
            return
        if a == "--version" or a == "-V":
            print("infer_train_rpc_server " + VERSION)
            return
        if a == "-m" or a == "--model":
            model_path = _next(arg_list, i)
            i += 1
        elif a == "--port":
            port = _next_int(arg_list, i)
            i += 1
        elif a == "--host":
            host = _next(arg_list, i)
            i += 1
        else:
            print("warning: unknown argument:", a)
        i += 1
    if model_path.byte_length() == 0:
        print("error: -m/--model is required (try --help)")
        return

    var ctx = load_gguf(model_path)
    var config = load_config(ctx)
    print("rpc server: model", model_path)
    print(
        "  arch:", arch_name(config.arch),
        "layers:", config.n_layers,
        "hidden:", config.hidden,
        "vocab:", config.vocab,
    )
    var listen_fd = tcp_listen(host, port)
    print(
        "  listening on", host, ":", port,
        "- waiting for master (the layer shard loads on INIT)",
    )
    var client_fd = tcp_accept(listen_fd)
    print("  master connected")

    # The protocol is strictly INIT-first: the master connects, assigns the
    # layer range, and only then starts forwarding tokens.
    var msg = recv_msg(client_fd)
    if Int(msg[0]) != Int(CMD_INIT):
        print("error: expected INIT as the first message")
        tcp_close(client_fd)
        tcp_close(listen_fd)
        return
    var lo = read_i32_le(msg, 1)
    var hi = read_i32_le(msg, 5)
    var ctx_len = read_i32_le(msg, 9)
    print("loading layers", lo, "..", hi - 1, "(ctx", ctx_len, ")")
    var shard = TransformerModel(
        config, ctx^, ctx_len, shard_lo=lo, shard_hi=hi, load_heads=False
    )
    print("shard ready: layers", lo, "..", hi - 1)
    var resp = List[UInt8]()
    resp.append(RPC_OK)
    append_i32(resp, config.hidden)
    send_msg(client_fd, resp)

    while True:
        var m: List[UInt8]
        try:
            m = recv_msg(client_fd)
        except:
            print("master disconnected")
            break
        var cmd = Int(m[0])
        if cmd == Int(CMD_PING):
            var r = List[UInt8]()
            r.append(RPC_OK)
            send_msg(client_fd, r)
        elif cmd == Int(CMD_FORWARD):
            var position = read_i32_le(m, 1)
            var body = List[UInt8]()
            for j in range(5, len(m)):
                body.append(m[j])
            var x = bytes_to_f16_tensor(body, config.hidden)
            var y = shard.forward_range(position, x)
            var r2 = List[UInt8]()
            r2.append(RPC_OK)
            var yb = f16_tensor_to_bytes(y)
            for b in yb:
                r2.append(b)
            send_msg(client_fd, r2)
        elif cmd == Int(CMD_RESET):
            shard.reset_cache()
            var r3 = List[UInt8]()
            r3.append(RPC_OK)
            send_msg(client_fd, r3)
        else:
            _err_reply(client_fd, "unknown command " + String(cmd))
    tcp_close(client_fd)
    tcp_close(listen_fd)
    print("rpc server: bye")


def _err_reply(fd: Int64, msg: String) raises:
    var resp = List[UInt8]()
    resp.append(RPC_ERR)
    var mb = msg.as_bytes()
    for i in range(len(mb)):
        resp.append(mb[i])
    resp.append(UInt8(0))
    send_msg(fd, resp)


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
