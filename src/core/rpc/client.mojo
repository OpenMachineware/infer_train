# core/rpc/client.mojo
#
# M8: the master-side RPC client.  One `RpcClient` is one connection to one
# worker (`infer_train_rpc_server` process); the CLI keeps a list of them
# and chains them per token for the `-sm layer` split.
#
# Protocol (see net.mojo for the framing):
#   INIT    [lo:i32][hi:i32][ctx_len:i32]
#           -> OK [hidden:i32] | ERR [msg]
#   FORWARD [position:i32][hidden state: fp16 LE bytes]
#           -> OK [hidden state] | ERR [msg]
#   RESET   -> OK | ERR [msg]
#   PING    -> OK | ERR [msg]

from .net import (
    tcp_connect,
    tcp_close,
    send_msg,
    recv_msg,
    append_i32,
    read_i32_le,
    f16_tensor_to_bytes,
    bytes_to_f16_tensor,
    rpc_err_string,
    CMD_INIT,
    CMD_FORWARD,
    CMD_RESET,
    CMD_PING,
    RPC_OK,
)
from ..tensor import Tensor


struct RpcClient(Movable):
    var fd: Int64
    var endpoint: String
    var hidden: Int

    def __init__(out self):
        self.fd = Int64(-1)
        self.endpoint = String("")
        self.hidden = 0

    def connect(mut self, host: String, port: Int) raises:
        self.fd = tcp_connect(host, port)
        self.endpoint = host + ":" + String(port)

    def call(mut self, payload: List[UInt8]) raises -> List[UInt8]:
        """One blocking request/response exchange."""
        send_msg(self.fd, payload)
        return recv_msg(self.fd)

    def ping(mut self) raises -> Bool:
        var payload = List[UInt8]()
        payload.append(UInt8(CMD_PING))
        var resp = self.call(payload)
        return len(resp) > 0 and resp[0] == RPC_OK

    def init_shard(mut self, lo: Int, hi: Int, ctx_len: Int) raises -> Int:
        """Ask the worker to load model layers [lo, hi); returns hidden."""
        var payload = List[UInt8]()
        payload.append(UInt8(CMD_INIT))
        append_i32(payload, lo)
        append_i32(payload, hi)
        append_i32(payload, ctx_len)
        var resp = self.call(payload)
        if len(resp) < 5 or resp[0] != RPC_OK:
            raise Error(
                "rpc init failed on " + self.endpoint + ": "
                + rpc_err_string(resp)
            )
        self.hidden = read_i32_le(resp, 1)
        return self.hidden

    def forward(
        mut self, position: Int, x: Tensor[DType.float16, 2]
    ) raises -> Tensor[DType.float16, 2]:
        """Run the worker's layer range on the hidden state at `position`."""
        var payload = List[UInt8]()
        payload.append(UInt8(CMD_FORWARD))
        append_i32(payload, position)
        var xb = f16_tensor_to_bytes(x)
        for b in xb:
            payload.append(b)
        var resp = self.call(payload)
        if len(resp) < 2 or resp[0] != RPC_OK:
            raise Error(
                "rpc forward failed on " + self.endpoint + ": "
                + rpc_err_string(resp)
            )
        # copy resp[1:] (Mojo 1.0: no List slicing)
        var body = List[UInt8]()
        for i in range(1, len(resp)):
            body.append(resp[i])
        return bytes_to_f16_tensor(body, self.hidden)

    def reset(mut self) raises:
        var payload = List[UInt8]()
        payload.append(UInt8(CMD_RESET))
        var resp = self.call(payload)
        if len(resp) < 1 or resp[0] != RPC_OK:
            raise Error(
                "rpc reset failed on " + self.endpoint + ": "
                + rpc_err_string(resp)
            )

    def close(mut self):
        tcp_close(self.fd)
        self.fd = Int64(-1)
