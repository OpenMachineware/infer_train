# core/rpc/net.mojo
#
# M8: the RPC transport - Mojo bindings over the C TCP helpers in
# tools/thread_pool.c (`it_tcp_*`), plus the message framing and the
# little serialization helpers shared by the master (`RpcClient`) and the
# worker (`infer_train_rpc_server`).
#
# Wire format: every message is a 4-byte little-endian payload length
# followed by the payload.  The payload starts with a 1-byte command tag
# (`CMD_*`) followed by command-specific arguments.  All integers are
# little-endian; the hidden state crosses the wire as raw fp16 bit
# patterns (2 bytes/element, lossless - the same values the engine keeps
# in memory, so a distributed run is numerically identical to a local one).
#
# Mojo 1.0's stdlib has no socket API, so the transport goes through
# `external_call` into the C runtime-helper library - the same pattern the
# thread pool and the file mmap use.

from ..thread_pool import _load_tp_library, _cstr
from ..tensor import Tensor, tensor_zeros
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.memory.unsafe import bitcast
from std.utils.static_tuple import StaticTuple
from std.collections import Span

# -- protocol command tags ----------------------------------------------------

comptime CMD_PING = UInt8(1)
comptime CMD_INIT = UInt8(2)
comptime CMD_FORWARD = UInt8(3)
comptime CMD_RESET = UInt8(4)

# Response status bytes.
comptime RPC_OK = UInt8(0)
comptime RPC_ERR = UInt8(1)


# -- raw socket helpers (C: it_tcp_*) -----------------------------------------


def tcp_listen(host: String, port: Int) raises -> Int64:
    """Bind + listen on `host:port` ('' / 0.0.0.0 = all interfaces)."""
    _load_tp_library()
    var fd = external_call[
        "it_tcp_listen",
        Int64,
        Pointer[UInt8, MutUntrackedOrigin],
        Int32,
    ](_cstr(host), Int32(port))
    if fd < 0:
        raise Error(
            "tcp_listen: cannot bind " + host + ":" + String(port)
        )
    return fd


def tcp_accept(listen_fd: Int64) raises -> Int64:
    """Blocking accept on a listening socket."""
    _load_tp_library()
    var fd = external_call["it_tcp_accept", Int64, Int64](listen_fd)
    if fd < 0:
        raise Error("tcp_accept: failed")
    return fd


def tcp_connect(host: String, port: Int) raises -> Int64:
    """Blocking connect to `host:port`."""
    _load_tp_library()
    var fd = external_call[
        "it_tcp_connect",
        Int64,
        Pointer[UInt8, MutUntrackedOrigin],
        Int32,
    ](_cstr(host), Int32(port))
    if fd < 0:
        raise Error(
            "tcp_connect: cannot connect to " + host + ":" + String(port)
        )
    return fd


def tcp_send(fd: Int64, data: List[UInt8]) raises:
    """Write all of `data` (the C side loops until the buffer is sent)."""
    if len(data) == 0:
        return
    _load_tp_library()
    var buf = unsafe_alloc[UInt8](len(data))
    for i in range(len(data)):
        buf.unsafe_store(i, data[i])
    var n = external_call[
        "it_tcp_send",
        Int64,
        Int64,
        Pointer[UInt8, MutUntrackedOrigin],
        Int64,
    ](fd, buf, Int64(len(data)))
    buf.unsafe_free()
    if n < 0:
        raise Error("tcp_send: connection broken")


def tcp_recv_some(
    fd: Int64,
    buf: Pointer[UInt8, MutUntrackedOrigin],
    cap: Int,
) raises -> Int:
    """One recv: bytes read, 0 = peer closed, raises on error."""
    _load_tp_library()
    var n = external_call[
        "it_tcp_recv",
        Int64,
        Int64,
        Pointer[UInt8, MutUntrackedOrigin],
        Int64,
    ](fd, buf, Int64(cap))
    if n < 0:
        raise Error("tcp_recv: connection broken")
    return Int(n)


def tcp_close(fd: Int64):
    if fd < 0:
        return
    _load_tp_library()
    _ = external_call["it_tcp_close", Int32, Int64](fd)


# -- framed message I/O --------------------------------------------------------


def send_msg(fd: Int64, payload: List[UInt8]) raises:
    """Send one framed message (4-byte LE length + payload)."""
    var frame = List[UInt8]()
    var n = len(payload)
    frame.append(UInt8(n & 0xFF))
    frame.append(UInt8((n >> 8) & 0xFF))
    frame.append(UInt8((n >> 16) & 0xFF))
    frame.append(UInt8((n >> 24) & 0xFF))
    for b in payload:
        frame.append(b)
    tcp_send(fd, frame)


def recv_msg(fd: Int64) raises -> List[UInt8]:
    """Receive one framed message; raises when the peer closes."""
    var hdr = unsafe_alloc[UInt8](4)
    var got = 0
    while got < 4:
        var n = tcp_recv_some(fd, hdr.unsafe_offset(got), 4 - got)
        if n == 0:
            hdr.unsafe_free()
            raise Error("recv_msg: connection closed")
        got += n
    var n = Int(hdr.unsafe_load[width=1](offset=0))
    n |= Int(hdr.unsafe_load[width=1](offset=1)) << 8
    n |= Int(hdr.unsafe_load[width=1](offset=2)) << 16
    n |= Int(hdr.unsafe_load[width=1](offset=3)) << 24
    hdr.unsafe_free()
    if n < 0 or n > (1 << 30):
        raise Error("recv_msg: bad length " + String(n))
    var buf = unsafe_alloc[UInt8](n if n > 0 else 1)
    var off = 0
    while off < n:
        var n2 = tcp_recv_some(fd, buf.unsafe_offset(off), n - off)
        if n2 == 0:
            buf.unsafe_free()
            raise Error("recv_msg: connection closed mid-message")
        off += n2
    var payload = List[UInt8]()
    for i in range(n):
        payload.append(buf.unsafe_load[width=1](offset=i))
    buf.unsafe_free()
    return payload^


# -- little-endian int helpers -------------------------------------------------


def i32_le(v: Int) -> List[UInt8]:
    """Encode a signed 32-bit value as 4 little-endian bytes."""
    var out = List[UInt8]()
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))
    return out^


def read_i32_le(data: List[UInt8], off: Int) -> Int:
    """Decode a signed 32-bit value from 4 little-endian bytes."""
    var v = Int(data[off])
    v |= Int(data[off + 1]) << 8
    v |= Int(data[off + 2]) << 16
    v |= Int(data[off + 3]) << 24
    if v >= 0x80000000:
        v -= 0x100000000
    return v


def append_i32(mut out: List[UInt8], v: Int):
    var b = i32_le(v)
    for x in b:
        out.append(x)


# -- fp16 hidden-state (de)serialization ---------------------------------------
#
# The fp16 -> int16 bitcast is a true type pun in this toolchain (verified:
# 1.5 -> 0x3E00, -2.25 -> 0xB400, lossless round-trips), so the wire bytes
# are exactly the in-memory bit patterns.


def f16_tensor_to_bytes(t: Tensor[DType.float16, 2]) -> List[UInt8]:
    """Raw little-endian fp16 bit patterns of a [1, n] tensor."""
    var n = t.numel()
    var out = List[UInt8]()
    for i in range(n):
        var bits = Int(bitcast[DType.int16](t.get(i))) & 0xFFFF
        out.append(UInt8(bits & 0xFF))
        out.append(UInt8((bits >> 8) & 0xFF))
    return out^


def bytes_to_f16_tensor(
    b: List[UInt8], hidden: Int
) -> Tensor[DType.float16, 2]:
    """Rebuild a [1, hidden] tensor from raw little-endian fp16 bytes."""
    var t = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, hidden))
    for i in range(hidden):
        var bits = Int(b[i * 2]) | (Int(b[i * 2 + 1]) << 8)
        t.set(
            i,
            Scalar[DType.float16](
                bitcast[DType.float16](Scalar[DType.int16](bits))
            ),
        )
    return t


# -- error-string helper -------------------------------------------------------


def rpc_err_string(resp: List[UInt8]) -> String:
    """Decode the trailing NUL-terminated message of an RPC_ERR response."""
    if len(resp) < 2:
        return String("unknown error")
    var n = 0
    var i = 1
    while i < len(resp) and resp[i] != UInt8(0):
        n += 1
        i += 1
    if n == 0:
        return String("unknown error")
    var buf = unsafe_alloc[UInt8](n)
    for j in range(n):
        buf.unsafe_store(j, resp[1 + j])
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=n)
    var s = String(unsafe_from_utf8=span)
    buf.unsafe_free()
    return s
