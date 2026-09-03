# core/http.mojo
#
# The it-server HTTP layer: a minimal HTTP/1.1 server over the `it_tcp_*`
# C helpers (the same transport core/rpc uses) serving the OpenAI-compatible
# surface the Python wrapper (python/infer_train/server.py) exposes:
#
#   GET  /health                   liveness probe
#   GET  /v1/models                available models
#   POST /v1/completions           text completion (+ "stream": true SSE)
#   POST /v1/chat/completions      chat completion (messages array)
#   POST /v1/finetune              inference-time fine-tune (synchronous)
#   GET  /v1/finetune/status       fine-tune jobs
#
# Design notes:
#   * Single-threaded by design (Mojo 1.0's stdlib has no thread API):
#     one connection at a time, `Connection: close` after every response.
#     That is all a verification/dev server needs; the Python wrapper
#     remains for concurrent production use.
#   * Request bodies are parsed with the streaming JsonParser (core/json)
#     - no DOM; unknown fields are skipped.
#   * API-key auth mirrors the Python wrapper: enabled when any key is
#     configured (--api-key flags, or the INFERTRAIN_API_KEY env var),
#     checked via `Authorization: Bearer <key>` or `x-api-key: <key>`.
#   * /v1/finetune runs synchronously (the single-threaded server cannot
#     queue background jobs) and records the job for /v1/finetune/status.

from src.core.rpc import (
    tcp_listen,
    tcp_accept,
    tcp_send,
    tcp_recv_some,
    tcp_close,
)
from src.core.json import JsonParser
from src.core.cli_common import GenState, basename
from src.core.thread_pool import now_ns
from src.runtime.inference import Model
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.collections import Span


# -- byte/string helpers ---------------------------------------------------------


def _append_string(mut out: List[UInt8], s: String):
    var b = s.as_bytes()
    for i in range(len(b)):
        out.append(b[i])


def _bytes_to_string(b: List[UInt8]) -> String:
    var n = len(b)
    if n == 0:
        return String("")
    var buf = unsafe_alloc[UInt8](n)
    for i in range(n):
        buf.unsafe_store(i, b[i])
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=n)
    var s = String(unsafe_from_utf8=span)
    buf.unsafe_free()
    return s


def _find_byte(s: String, byte: UInt8) -> Int:
    var b = s.as_bytes()
    for i in range(len(b)):
        if b[i] == byte:
            return i
    return -1


def _slice_str(s: String, start: Int, end: Int) -> String:
    var n = end - start
    if n <= 0:
        return String("")
    var src = s.as_bytes()
    var buf = unsafe_alloc[UInt8](n)
    for i in range(n):
        buf.unsafe_store(i, src[start + i])
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=buf, length=n)
    var out = String(unsafe_from_utf8=span)
    buf.unsafe_free()
    return out


def _to_lower(s: String) -> String:
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c >= 65 and c <= 90:
            out.append(UInt8(c + 32))
        else:
            out.append(b[i])
    return _bytes_to_string(out)


def _trim(s: String) -> String:
    var b = s.as_bytes()
    var lo = 0
    var hi = len(b)
    while lo < hi and (b[lo] == UInt8(32) or b[lo] == UInt8(9)):
        lo += 1
    while hi > lo and (b[hi - 1] == UInt8(32) or b[hi - 1] == UInt8(9)):
        hi -= 1
    return _slice_str(s, lo, hi)


def _atoi(s: String) -> Int:
    var v = 0
    var neg = False
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c == 45:
            neg = True
        elif c >= 48 and c <= 57:
            v = v * 10 + (c - 48)
    return -v if neg else v


def _split_crlf(s: String) -> List[String]:
    var lines = List[String]()
    var b = s.as_bytes()
    var start = 0
    for i in range(len(b) - 1):
        if b[i] == UInt8(13) and b[i + 1] == UInt8(10):
            lines.append(_slice_str(s, start, i))
            start = i + 2
    lines.append(_slice_str(s, start, len(b)))
    return lines^


def _hex2(c: Int) -> String:
    var hi = c // 16
    var lo = c % 16
    var h = String("0123456789abcdef")
    return _slice_str(h, hi, hi + 1) + _slice_str(h, lo, lo + 1)


def json_escape(s: String) -> String:
    """Escape a string for embedding in a JSON string literal."""
    var out = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c == 34:  # '"'
            _append_string(out, String("\\\""))
        elif c == 92:  # '\\'
            _append_string(out, String("\\\\"))
        elif c == 10:
            _append_string(out, String("\\n"))
        elif c == 13:
            _append_string(out, String("\\r"))
        elif c == 9:
            _append_string(out, String("\\t"))
        elif c < 32:
            _append_string(out, String("\\u00"))
            _append_string(out, _hex2(c))
        else:
            out.append(b[i])
    return _bytes_to_string(out)


# -- HTTP/1.1 request reading ------------------------------------------------------


struct HttpRequest(Movable):
    var method: String
    var path: String
    var content_length: Int
    var authorization: String
    var x_api_key: String
    var body: List[UInt8]

    def __init__(out self):
        self.method = String("")
        self.path = String("")
        self.content_length = 0
        self.authorization = String("")
        self.x_api_key = String("")
        self.body = List[UInt8]()


def _find_header_end(buf: List[UInt8]) -> Int:
    """Offset just past the first `\r\n\r\n`, or -1 when incomplete."""
    if len(buf) < 4:
        return -1
    for i in range(len(buf) - 3):
        if (
            buf[i] == UInt8(13)
            and buf[i + 1] == UInt8(10)
            and buf[i + 2] == UInt8(13)
            and buf[i + 3] == UInt8(10)
        ):
            return i + 4
    return -1


def read_request(fd: Int64) raises -> HttpRequest:
    """Read one HTTP request (headers + Content-Length body bytes)."""
    var buf = List[UInt8]()
    var tmp = unsafe_alloc[UInt8](8192)
    var header_end = -1
    while header_end < 0:
        var n = tcp_recv_some(fd, tmp, 8192)
        if n == 0:
            tmp.unsafe_free()
            raise Error("client closed the connection")
        for i in range(n):
            buf.append(tmp.unsafe_load[width=1](offset=i))
        header_end = _find_header_end(buf)
        if len(buf) > 1048576:
            tmp.unsafe_free()
            raise Error("request headers too large")
    tmp.unsafe_free()

    # Request line: METHOD SP TARGET SP HTTP/1.x
    var head = List[UInt8]()
    for i in range(header_end - 2):
        head.append(buf[i])
    var head_s = _bytes_to_string(head)
    var sp1 = _find_byte(head_s, UInt8(32))
    if sp1 <= 0:
        raise Error("bad request line")
    var method = _slice_str(head_s, 0, sp1)
    var rest = _slice_str(head_s, sp1 + 1, head_s.byte_length())
    var sp2 = _find_byte(rest, UInt8(32))
    var target = rest if sp2 < 0 else _slice_str(rest, 0, sp2)
    var q = _find_byte(target, UInt8(63))  # strip ?query
    if q >= 0:
        target = _slice_str(target, 0, q)

    var req = HttpRequest()
    req.method = method
    req.path = target
    # Headers (the lines after the request line).
    var lines = _split_crlf(head_s)
    for i in range(1, len(lines)):
        var line = lines[i]
        var colon = _find_byte(line, UInt8(58))
        if colon <= 0:
            continue
        var name = _to_lower(_slice_str(line, 0, colon))
        var value = _trim(_slice_str(line, colon + 1, line.byte_length()))
        if name == "content-length":
            req.content_length = _atoi(value)
        elif name == "authorization":
            req.authorization = value
        elif name == "x-api-key":
            req.x_api_key = value
    # Body: whatever already arrived plus the remainder.
    var body = List[UInt8]()
    var have = len(buf) - header_end
    for i in range(have):
        body.append(buf[header_end + i])
    var tmp2 = unsafe_alloc[UInt8](8192)
    while len(body) < req.content_length:
        var n2 = tcp_recv_some(fd, tmp2, 8192)
        if n2 == 0:
            tmp2.unsafe_free()
            raise Error("client closed mid-body")
        for i in range(n2):
            body.append(tmp2.unsafe_load[width=1](offset=i))
    tmp2.unsafe_free()
    req.body = body^
    return req^


# -- HTTP/1.1 responses -------------------------------------------------------------


def _status_text(status: Int) -> String:
    if status == 200:
        return String("OK")
    if status == 400:
        return String("Bad Request")
    if status == 401:
        return String("Unauthorized")
    if status == 404:
        return String("Not Found")
    return String("Error")


def send_json(fd: Int64, status: Int, body: String) raises:
    var out = List[UInt8]()
    _append_string(out, String("HTTP/1.1 "))
    _append_string(out, String(status))
    _append_string(out, String(" "))
    _append_string(out, _status_text(status))
    _append_string(out, String("\r\n"))
    _append_string(out, String("Content-Type: application/json\r\n"))
    _append_string(out, String("Content-Length: "))
    _append_string(out, String(body.byte_length()))
    _append_string(out, String("\r\n"))
    _append_string(out, String("Connection: close\r\n\r\n"))
    _append_string(out, body)
    tcp_send(fd, out)


def send_error(fd: Int64, status: Int, message: String) raises:
    var w = JsonWriter()
    w.append("{\"error\": {\"message\": \"")
    w.esc(message)
    w.append("\", \"type\": \"invalid_request_error\"}}")
    send_json(fd, status, w.done())


def send_sse_headers(fd: Int64) raises:
    var out = List[UInt8]()
    _append_string(out, String("HTTP/1.1 200 OK\r\n"))
    _append_string(out, String("Content-Type: text/event-stream\r\n"))
    _append_string(out, String("Cache-Control: no-cache\r\n"))
    _append_string(out, String("Connection: close\r\n\r\n"))
    tcp_send(fd, out)


def send_sse_event(fd: Int64, data: String) raises:
    var out = List[UInt8]()
    _append_string(out, String("data: "))
    _append_string(out, data)
    _append_string(out, String("\n\n"))
    tcp_send(fd, out)


def send_sse_done(fd: Int64) raises:
    var out = List[UInt8]()
    _append_string(out, String("data: [DONE]\n\n"))
    tcp_send(fd, out)


# -- JSON response writer -------------------------------------------------------------


struct JsonWriter(Movable):
    """Accumulates a JSON document as bytes (no String += at runtime)."""

    var out: List[UInt8]

    def __init__(out self):
        self.out = List[UInt8]()

    def append(mut self, s: String):
        _append_string(self.out, s)

    def esc(mut self, s: String):
        _append_string(self.out, json_escape(s))

    def int(mut self, v: Int):
        _append_string(self.out, String(v))

    def f32(mut self, v: Float32):
        _append_string(self.out, String(v))

    def done(self) -> String:
        return _bytes_to_string(self.out)


# -- request-body parsing (OpenAI completion / finetune fields) ----------------------


struct RequestParams(Movable):
    var prompt: String
    var max_tokens: Int
    var temperature: Float32
    var top_p: Float32
    var top_k: Int
    var seed: Int  # -1 = unset (random)
    var stream: Bool
    var input_text: String
    var target_text: String
    var lr: Float32

    def __init__(out self):
        self.prompt = String("")
        self.max_tokens = 32
        self.temperature = Float32(0.6)
        self.top_p = Float32(0.95)
        self.top_k = 40
        self.seed = -1
        self.stream = False
        self.input_text = String("")
        self.target_text = String("")
        self.lr = Float32(1e-5)


def _parse_messages(mut parser: JsonParser) raises -> String:
    """Flatten a messages array to a prompt (mirrors the Python wrapper's
    `_prompt_of`: every content gets a newline, joined with newlines,
    stripped)."""
    parser.skip_ws()
    parser.expect_byte(UInt8(91))  # '['
    var parts = List[String]()
    parser.skip_ws()
    if parser.peek() == UInt8(93):  # ']'
        parser.advance()
        return String("")
    while True:
        parser.skip_ws()
        parser.expect_byte(UInt8(123))  # '{'
        var content = String("")
        while True:
            parser.skip_ws()
            var c = parser.peek()
            if c == UInt8(125):  # '}'
                parser.advance()
                break
            var key = parser.parse_string()
            parser.skip_ws()
            parser.expect_byte(UInt8(58))  # ':'
            if key == "content":
                content = parser.parse_string()
            else:
                parser.skip_value()
            parser.skip_ws()
            var c2 = parser.peek()
            if c2 == UInt8(44):  # ','
                parser.advance()
                continue
            if c2 == UInt8(125):  # '}'
                parser.advance()
                break
            raise Error("json: bad message object")
        parts.append(content + String("\n"))
        parser.skip_ws()
        var c3 = parser.peek()
        if c3 == UInt8(44):  # ','
            parser.advance()
            continue
        if c3 == UInt8(93):  # ']'
            parser.advance()
            break
        raise Error("json: bad messages array")
    var joined = String("")
    for i in range(len(parts)):
        if i > 0:
            joined = joined + String("\n")
        joined = joined + parts[i]
    return _trim(joined)


def parse_request(body: List[UInt8]) raises -> RequestParams:
    """Extract the request fields; unknown fields are skipped."""
    var p = RequestParams()
    if len(body) == 0:
        return p^
    var buf = unsafe_alloc[UInt8](len(body))
    for i in range(len(body)):
        buf.unsafe_store(i, body[i])
    var parser = JsonParser(buf, len(body))
    parser.skip_ws()
    parser.expect_byte(UInt8(123))  # '{'
    while True:
        parser.skip_ws()
        var c = parser.peek()
        if c == UInt8(125):  # '}'
            break
        if c != UInt8(34):
            raise Error("json: expected object key")
        var key = parser.parse_string()
        parser.skip_ws()
        parser.expect_byte(UInt8(58))  # ':'
        if key == "prompt":
            p.prompt = parser.parse_string()
        elif key == "messages":
            p.prompt = _parse_messages(parser)
        elif key == "max_tokens":
            p.max_tokens = parser.parse_int_raw()
        elif key == "temperature":
            p.temperature = Float32(parser.parse_number().as_float())
        elif key == "top_p":
            p.top_p = Float32(parser.parse_number().as_float())
        elif key == "top_k":
            p.top_k = parser.parse_int_raw()
        elif key == "seed":
            if parser.peek() == UInt8(110):  # 'n' of null
                parser.advance()
                parser.advance()
                parser.advance()
                parser.advance()
                p.seed = -1
            else:
                p.seed = parser.parse_int_raw()
        elif key == "stream":
            p.stream = parser.read_bool_raw()
        elif key == "input" or key == "input_text":
            p.input_text = parser.parse_string()
        elif key == "target" or key == "target_text":
            p.target_text = parser.parse_string()
        elif key == "lr":
            p.lr = Float32(parser.parse_number().as_float())
        else:
            parser.skip_value()
        parser.skip_ws()
        var c2 = parser.peek()
        if c2 == UInt8(44):  # ','
            parser.advance()
            continue
        if c2 == UInt8(125):  # '}'
            break
        raise Error("json: expected ',' or '}'")
    buf.unsafe_free()
    return p^


# -- inference-time fine-tuning (LoRA-style output-head adapter) ---------------------
#
# Same math as the C-API infer_train_finetune_* (bindings): a persistent
# fp32 copy of the output head + AdamW moments; each step computes the
# adapter logits and one update toward the target token; the fp16 head in
# the model is re-synced after every update so subsequent inference sees
# the adapted weights immediately.


def _exp_f32(x: Float32) -> Float32:
    from std.math import exp

    return exp(x)


def _log_f32(x: Float32) -> Float32:
    from std.math import log

    return log(x)


def _pow_f32(b: Float32, e: Float32) -> Float32:
    from std.math import pow

    return pow(b, e)


def _sqrt_f32(x: Float32) -> Float32:
    from std.math import sqrt

    return sqrt(x)


struct FinetuneSession(Movable):
    var w: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var m: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var v: Pointer[Scalar[DType.float32], MutUntrackedOrigin]
    var vocab: Int
    var hidden: Int
    var t: Int
    var lr: Float32

    def __init__(
        out self,
        model_p: Pointer[Model, MutUntrackedOrigin],
        lr: Float32,
    ) raises:
        var vocab = model_p[0].transformer.config.vocab
        var hidden = model_p[0].transformer.config.hidden
        self.w = unsafe_alloc[Scalar[DType.float32]](vocab * hidden)
        self.m = unsafe_alloc[Scalar[DType.float32]](vocab * hidden)
        self.v = unsafe_alloc[Scalar[DType.float32]](vocab * hidden)
        self.vocab = vocab
        self.hidden = hidden
        self.t = 0
        self.lr = lr
        # Seed the adapter from the model's current head (fp16 -> fp32).
        var head = model_p[0].transformer.params.output_w
        for i in range(vocab * hidden):
            self.w.unsafe_store(
                i, Scalar[DType.float32](Float32(head.get(i)))
            )
            self.m.unsafe_store(i, Scalar[DType.float32](0))
            self.v.unsafe_store(i, Scalar[DType.float32](0))

    def __deinit__(deinit self):
        self.w.unsafe_free()
        self.m.unsafe_free()
        self.v.unsafe_free()

    def step(
        mut self,
        model_p: Pointer[Model, MutUntrackedOrigin],
        token: Int,
        position: Int,
        target: Int,
    ) raises -> Float32:
        """One forward step; one AdamW update when target >= 0."""
        var h = model_p[0].transformer.forward_hidden(token, position)
        if target < 0:
            return Float32(0)  # forward-only: cache/SSM state advanced
        var vocab = self.vocab
        var hidden = self.hidden
        # Logits + stable softmax over the fp32 adapter head.
        var mx = Float32(-3.0e38)
        var lsum = Float32(0)
        var probs = unsafe_alloc[Scalar[DType.float32]](vocab)
        for i in range(vocab):
            var acc = Float32(0)
            for j in range(hidden):
                acc += Float32(self.w.unsafe_load(offset=i * hidden + j)) * Float32(
                    h.get(j)
                )
            probs.unsafe_store(i, Scalar[DType.float32](acc))
            if acc > mx:
                mx = acc
        for i in range(vocab):
            var e = _exp_f32(Float32(probs.unsafe_load(offset=i)) - mx)
            probs.unsafe_store(i, Scalar[DType.float32](e))
            lsum += e
        var inv = Float32(1.0) / lsum
        var loss = Float32(0)
        for i in range(vocab):
            var p = Float32(probs.unsafe_load(offset=i)) * inv
            if p < Float32(1e-9):
                p = Float32(1e-9)  # clamp: avoid log(0) -> inf
            probs.unsafe_store(i, Scalar[DType.float32](p))
            if i == target:
                loss = -_log_f32(p)
        # AdamW update: dW[i, j] = (p_i - 1[i==target]) * h[j]
        self.t += 1
        var b1 = Float32(0.9)
        var b2 = Float32(0.999)
        var t = Float32(self.t)
        var bc1 = Float32(1.0) - _pow_f32(b1, t)
        var bc2 = Float32(1.0) - _pow_f32(b2, t)
        if bc1 < Float32(1e-12):
            bc1 = Float32(1e-12)
        if bc2 < Float32(1e-12):
            bc2 = Float32(1e-12)
        var eps = Float32(1e-8)
        for i in range(vocab):
            var label = Float32(1.0) if i == target else Float32(0)
            var base = i * hidden
            for j in range(hidden):
                var g = (
                    Float32(probs.unsafe_load(offset=i)) - label
                ) * Float32(h.get(j))
                var m_v = b1 * Float32(self.m.unsafe_load(offset=base + j)) + (
                    Float32(1.0) - b1
                ) * g
                var v_v = b2 * Float32(self.v.unsafe_load(offset=base + j)) + (
                    Float32(1.0) - b2
                ) * g * g
                self.m.unsafe_store(base + j, Scalar[DType.float32](m_v))
                self.v.unsafe_store(base + j, Scalar[DType.float32](v_v))
                var update = (m_v / bc1) / (_sqrt_f32(v_v / bc2) + eps)
                self.w.unsafe_store(
                    base + j,
                    Scalar[DType.float32](
                        Float32(self.w.unsafe_load(offset=i * hidden + j))
                        - self.lr * update
                    ),
                )
        # Sync the fp16 head back into the model (inference sees it now).
        var head = model_p[0].transformer.params.output_w
        for i in range(vocab * hidden):
            head.set(
                i,
                Scalar[DType.float16](
                    Float32(self.w.unsafe_load(offset=i))
                ),
            )
        probs.unsafe_free()
        return loss


# -- fine-tune job bookkeeping --------------------------------------------------------


struct FinetuneJob(Movable):
    var id: String
    var status: String
    var steps: Int
    var loss: Float32
    var has_loss: Bool
    var losses: List[Float32]
    var error: String

    def __init__(out self):
        self.id = String("")
        self.status = String("")
        self.steps = 0
        self.loss = Float32(0)
        self.has_loss = False
        self.losses = List[Float32]()
        self.error = String("")

    def to_json(self) -> String:
        var w = JsonWriter()
        w.append("{\"id\": \"")
        w.esc(self.id)
        w.append("\", \"status\": \"")
        w.esc(self.status)
        w.append("\", \"steps\": ")
        w.int(self.steps)
        if self.has_loss:
            w.append(", \"loss\": ")
            w.f32(self.loss)
        w.append(", \"losses\": [")
        for i in range(len(self.losses)):
            if i > 0:
                w.append(String(", "))
            w.f32(self.losses[i])
        w.append("]")
        if self.error.byte_length() > 0:
            w.append(", \"error\": \"")
            w.esc(self.error)
            w.append("\"")
        w.append("}")
        return w.done()


# -- the server ------------------------------------------------------------------------


struct ServerState(Movable):
    var model_p: Pointer[Model, MutUntrackedOrigin]
    var model_name: String  # --alias, or the model file's basename
    var model_path: String
    var api_keys: List[String]
    var jobs: List[FinetuneJob]
    var job_seq: Int
    # Sampling defaults for requests that omit the fields.
    var temperature: Float32
    var top_p: Float32
    var top_k: Int
    var repeat_penalty: Float32

    def __init__(
        out self,
        model_p: Pointer[Model, MutUntrackedOrigin],
        model_name: String,
        model_path: String,
        api_keys: List[String],
        temperature: Float32,
        top_p: Float32,
        top_k: Int,
        repeat_penalty: Float32,
    ):
        self.model_p = model_p
        self.model_name = model_name
        self.model_path = model_path
        self.api_keys = api_keys.copy()
        self.jobs = List[FinetuneJob]()
        self.job_seq = 0
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

    def run(mut self, host: String, port: Int) raises:
        var listen_fd = tcp_listen(host, port)
        print("it-server: serving on http://" + host + ":" + String(port))
        print("  model: " + self.model_path + " (name: " + self.model_name + ")")
        print("  endpoints: /health /v1/models /v1/completions")
        print("             /v1/chat/completions /v1/finetune /v1/finetune/status")
        while True:
            var fd = tcp_accept(listen_fd)
            try:
                self.handle_connection(fd)
            except:
                print("it-server: connection error (ignored)")
            tcp_close(fd)

    def _auth_ok(self, authorization: String, x_api_key: String) -> Bool:
        if len(self.api_keys) == 0:
            return True
        var bearer = String("")
        if authorization.byte_length() > 7:
            var prefix = authorization.as_bytes()
            var ok = True
            var want = String("Bearer ")
            var wb = want.as_bytes()
            for i in range(7):
                if prefix[i] != wb[i]:
                    ok = False
            if ok:
                bearer = _slice_str(authorization, 7, authorization.byte_length())
        for i in range(len(self.api_keys)):
            if (
                bearer.byte_length() > 0
                and self.api_keys[i] == bearer
            ) or (
                x_api_key.byte_length() > 0
                and self.api_keys[i] == x_api_key
            ):
                return True
        return False

    def handle_connection(mut self, fd: Int64) raises:
        var req = read_request(fd)
        if not self._auth_ok(req.authorization, req.x_api_key):
            send_error(fd, 401, "invalid API key")
            return
        if req.method == "GET" and req.path == "/health":
            var w = JsonWriter()
            w.append("{\"status\": \"ok\", \"model\": \"")
            w.esc(self.model_name)
            w.append("\"}")
            send_json(fd, 200, w.done())
        elif req.method == "GET" and req.path == "/v1/models":
            var w = JsonWriter()
            w.append(
                "{\"object\": \"list\", \"data\": [{\"id\": \""
            )
            w.esc(self.model_name)
            w.append(
                "\", \"object\": \"model\", \"created\": 0,"
                " \"owned_by\": \"infer_train\"}]}"
            )
            send_json(fd, 200, w.done())
        elif req.method == "GET" and req.path == "/v1/finetune/status":
            var w = JsonWriter()
            w.append("{\"object\": \"list\", \"data\": [")
            for i in range(len(self.jobs)):
                if i > 0:
                    w.append(String(", "))
                w.append(self.jobs[i].to_json())
            w.append("]}")
            send_json(fd, 200, w.done())
        elif req.method == "POST" and req.path == "/v1/completions":
            self._completions(fd, req, False)
        elif req.method == "POST" and req.path == "/v1/chat/completions":
            self._completions(fd, req, True)
        elif req.method == "POST" and req.path == "/v1/finetune":
            self._finetune(fd, req)
        else:
            send_error(fd, 404, "not found")

    def _completions(mut self, fd: Int64, req: HttpRequest, chat: Bool) raises:
        var params = parse_request(req.body)
        if params.prompt.byte_length() == 0:
            send_error(fd, 400, "prompt is required")
            return
        if params.max_tokens < 1:
            params.max_tokens = 1
        self.model_p[0].transformer.reset_cache()
        var created = now_ns() / 1000000
        var cmpl_id = (
            ("chatcmpl-" if chat else "cmpl-") + String(created)
        )
        if params.stream:
            self._completions_sse(fd, cmpl_id, created, params)
            return
        var g = GenState(
            self.model_p,
            params.max_tokens,
            params.temperature,
            params.top_p,
            params.top_k,
            self.repeat_penalty,
            params.seed,
        )
        g.prefill(params.prompt)
        var finish = "length"
        while not g.done:
            var t = g.next_token()
            if t < 0:
                finish = "stop"
                break
        var text = g.decode_all()
        var n = len(g.generated)
        var w = JsonWriter()
        w.append("{\"id\": \"")
        w.esc(cmpl_id)
        if chat:
            w.append(
                "\", \"object\": \"chat.completion\", \"created\": "
            )
            w.int(created)
            w.append(", \"model\": \"")
            w.esc(self.model_name)
            w.append(
                "\", \"choices\": [{\"index\": 0, \"message\": "
                "{\"role\": \"assistant\", \"content\": \""
            )
            w.esc(text)
            w.append("\"}, \"finish_reason\": \"")
            w.append(finish)
            w.append(
                "\"}], \"usage\": {\"prompt_tokens\": 0,"
                " \"completion_tokens\": 0, \"total_tokens\": 0}}"
            )
        else:
            w.append(
                "\", \"object\": \"text_completion\", \"created\": "
            )
            w.int(created)
            w.append(", \"model\": \"")
            w.esc(self.model_name)
            w.append("\", \"choices\": [{\"text\": \"")
            w.esc(text)
            w.append("\", \"index\": 0, \"finish_reason\": \"")
            w.append(finish)
            w.append(
                "\"}], \"usage\": {\"prompt_tokens\": 0,"
                " \"completion_tokens\": "
            )
            w.int(n)
            w.append(", \"total_tokens\": ")
            w.int(n)
            w.append("}}")
        send_json(fd, 200, w.done())

    def _completions_sse(
        mut self,
        fd: Int64,
        cmpl_id: String,
        created: Int,
        params: RequestParams,
    ) raises:
        send_sse_headers(fd)
        var g = GenState(
            self.model_p,
            params.max_tokens,
            params.temperature,
            params.top_p,
            params.top_k,
            self.repeat_penalty,
            params.seed,
        )
        g.prefill(params.prompt)
        var finish = "length"
        while not g.done:
            var t = g.next_token()
            if t < 0:
                finish = "stop"
                break
            var piece = g.decode_token(t)
            var w = JsonWriter()
            w.append("{\"id\": \"")
            w.esc(cmpl_id)
            w.append(
                "\", \"object\": \"text_completion\", \"created\": "
            )
            w.int(created)
            w.append(", \"model\": \"")
            w.esc(self.model_name)
            w.append("\", \"choices\": [{\"text\": \"")
            w.esc(piece)
            w.append("\", \"index\": 0, \"finish_reason\": null}]}")
            send_sse_event(fd, w.done())
        var w2 = JsonWriter()
        w2.append("{\"id\": \"")
        w2.esc(cmpl_id)
        w2.append(
            "\", \"object\": \"text_completion\", \"created\": "
        )
        w2.int(created)
        w2.append(", \"model\": \"")
        w2.esc(self.model_name)
        w2.append(
            "\", \"choices\": [{\"text\": \"\", \"index\": 0,"
            " \"finish_reason\": \""
        )
        w2.append(finish)
        w2.append("\"}]}")
        send_sse_event(fd, w2.done())
        send_sse_done(fd)

    def _finetune(mut self, fd: Int64, req: HttpRequest) raises:
        var params = parse_request(req.body)
        if (
            params.input_text.byte_length() == 0
            or params.target_text.byte_length() == 0
        ):
            send_error(
                fd, 400, "finetune needs 'input' and 'target' fields"
            )
            return
        var job = FinetuneJob()
        self.job_seq += 1
        job.id = "ft-" + String(now_ns() / 1000000) + "-" + String(self.job_seq)
        job.status = "running"
        try:
            self.model_p[0].transformer.reset_cache()
            var prompt_tokens = self.model_p[0].tokenizer.encode_with_bos(
                params.input_text
            )
            var target_tokens = self.model_p[0].tokenizer.encode_with_bos(
                params.target_text
            )
            var eos = self.model_p[0].tokenizer.eos_id()
            target_tokens.append(eos)
            var session = FinetuneSession(self.model_p, params.lr)
            var pos = 0
            for i in range(len(prompt_tokens)):
                session.step(self.model_p, prompt_tokens[i], pos, -1)
                pos += 1
            for i in range(len(target_tokens)):
                var target = (
                    target_tokens[i + 1]
                    if i + 1 < len(target_tokens)
                    else eos
                )
                var loss = session.step(
                    self.model_p, target_tokens[i], pos, target
                )
                pos += 1
                job.losses.append(loss)
            job.status = "done"
            job.steps = len(job.losses)
            if len(job.losses) > 0:
                job.loss = job.losses[len(job.losses) - 1]
                job.has_loss = True
        except:
            job.status = "failed"
            job.error = "finetune failed"
        # Build the response from the job *before* moving it into the
        # job list (FinetuneJob is Movable, not copyable).
        var w = JsonWriter()
        w.append("{\"id\": \"")
        w.esc(job.id)
        w.append("\", \"status\": \"")
        w.esc(job.status)
        w.append("\", \"steps\": ")
        w.int(job.steps)
        if job.has_loss:
            w.append(", \"loss\": ")
            w.f32(job.loss)
        w.append(", \"losses\": [")
        for i in range(len(job.losses)):
            if i > 0:
                w.append(String(", "))
            w.f32(job.losses[i])
        w.append("]")
        if job.error.byte_length() > 0:
            w.append(", \"error\": \"")
            w.esc(job.error)
            w.append("\"")
        w.append("}")
        var resp = w.done()
        self.jobs.append(job^)
        send_json(fd, 200, resp)
