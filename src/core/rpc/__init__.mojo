# core/rpc/__init__.mojo
#
# M8: multi-process / multi-machine RPC layer (llama.cpp-style).
#
# One generic transport (plain TCP, length-prefixed messages) plus a
# layer-split distribution strategy (`-sm layer`): each `--rpc HOST:PORT`
# endpoint is a worker process (`it-rpc-server`) that owns a contiguous
# range of transformer layers and their KV/SSM state; the master (it-cli)
# keeps the embedding + output head and chains the workers per token.
# The protocol is deliberately small (INIT / FORWARD / RESET / PING) so a
# second distribution strategy (row-parallel, `-sm row`) can be added
# later on the same transport.
#
# Public surface:
#   * `RpcClient`  - master-side connection to one worker
#   * `tcp_listen` / `tcp_accept` / `tcp_connect` / `tcp_send` /
#     `tcp_recv_some` / `tcp_close` - the raw socket helpers
#   * `send_msg` / `recv_msg` - the framed message I/O
#   * `CMD_*` - protocol command tags

from .net import (
    tcp_listen,
    tcp_accept,
    tcp_connect,
    tcp_send,
    tcp_recv_some,
    tcp_close,
    send_msg,
    recv_msg,
    i32_le,
    read_i32_le,
    append_i32,
    f16_tensor_to_bytes,
    bytes_to_f16_tensor,
    rpc_err_string,
    CMD_PING,
    CMD_INIT,
    CMD_FORWARD,
    CMD_RESET,
    RPC_OK,
    RPC_ERR,
)
from .client import RpcClient
