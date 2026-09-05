# core/json.mojo
#
# A small hand-rolled JSON parser for Mojo 1.0.
#
# Mojo 1.0's standard library ships no `std.json`, and `tokenizer.json`
# (7 MB, 150k+ entries) is the largest file the runtime must read.  The
# parser therefore works directly over the mapped byte buffer and offers two
# entry points:
#
#   * `JsonParser` streaming primitives (`parse_string`, `parse_int_raw`,
#     `skip_value`, ...) used by the tokenizer to pull the vocab/merges
#     tables out of tokenizer.json without materializing a DOM; string keys
#     without escapes are built as zero-copy spans of the mapped file.
#   * `parse_json_flat_file(path)` which reads a *flat* JSON object (scalar
#     values only - exactly the shape of config.json / generation_config.json)
#     into a `Dict[String, JsonScalar]`.
#
# There is deliberately no recursive JSON DOM: a `JsonValue` containing
# `List[JsonValue]` would be a recursive type, which requires pointer
# indirection, and pointer-held containers misbehave at runtime in this
# Mojo 1.0 toolchain (a `Pointer[Dict]` subscript assignment hangs), so the
# design avoids that pattern altogether.
#
# The parser handles the JSON subset used by HuggingFace tokenizer/config
# files: objects, arrays, strings (incl. \uXXXX escapes and surrogate
# pairs), integers, floats (incl. exponents), booleans and null.

from .utils import unimplemented
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.collections import Span

comptime JSON_NULL = 0
comptime JSON_BOOL = 1
comptime JSON_INT = 2
comptime JSON_FLOAT = 3
comptime JSON_STR = 4


struct JsonScalar(Copyable, Movable):
    """A flat JSON scalar (no containers); see the module docstring."""

    var kind: Int8
    var bool_val: Bool
    var int_val: Int
    var float_val: Float64
    var str_val: String

    def __init__(out self):
        self.kind = JSON_NULL
        self.bool_val = False
        self.int_val = 0
        self.float_val = 0
        self.str_val = String("")

    @staticmethod
    def of_bool(value: Bool) -> Self:
        var v = Self()
        v.kind = JSON_BOOL
        v.bool_val = value
        return v^

    @staticmethod
    def of_int(value: Int) -> Self:
        var v = Self()
        v.kind = JSON_INT
        v.int_val = value
        return v^

    @staticmethod
    def of_float(value: Float64) -> Self:
        var v = Self()
        v.kind = JSON_FLOAT
        v.float_val = value
        return v^

    @staticmethod
    def of_str(var value: String) -> Self:
        var v = Self()
        v.kind = JSON_STR
        v.str_val = value^
        return v^

    def as_int(self) -> Int:
        if self.kind == JSON_INT:
            return self.int_val
        if self.kind == JSON_FLOAT:
            return Int(self.float_val)
        if self.kind == JSON_BOOL:
            return 1 if self.bool_val else 0
        return 0

    def as_float(self) -> Float64:
        if self.kind == JSON_FLOAT:
            return self.float_val
        if self.kind == JSON_INT:
            return Float64(self.int_val)
        return 0

    def as_bool(self) -> Bool:
        if self.kind == JSON_BOOL:
            return self.bool_val
        return False

    def as_str(self) -> String:
        if self.kind == JSON_STR:
            return self.str_val
        return String("")


def utf8_encode(
    cp: Int, buf: Pointer[UInt8, MutUntrackedOrigin], mut idx: Int
) -> Int:
    """Append the UTF-8 encoding of codepoint `cp`; returns the new index."""
    if cp < 0x80:
        buf.unsafe_store(idx, UInt8(cp))
        return idx + 1
    if cp < 0x800:
        buf.unsafe_store(idx, UInt8(0xC0 | (cp >> 6)))
        buf.unsafe_store(idx + 1, UInt8(0x80 | (cp & 0x3F)))
        return idx + 2
    if cp < 0x10000:
        buf.unsafe_store(idx, UInt8(0xE0 | (cp >> 12)))
        buf.unsafe_store(idx + 1, UInt8(0x80 | ((cp >> 6) & 0x3F)))
        buf.unsafe_store(idx + 2, UInt8(0x80 | (cp & 0x3F)))
        return idx + 3
    buf.unsafe_store(idx, UInt8(0xF0 | (cp >> 18)))
    buf.unsafe_store(idx + 1, UInt8(0x80 | ((cp >> 12) & 0x3F)))
    buf.unsafe_store(idx + 2, UInt8(0x80 | ((cp >> 6) & 0x3F)))
    buf.unsafe_store(idx + 3, UInt8(0x80 | (cp & 0x3F)))
    return idx + 4


def hex_value(byte: UInt8) -> Int:
    if byte >= UInt8(48) and byte <= UInt8(57):  # '0'-'9'
        return Int(byte) - 48
    if byte >= UInt8(97) and byte <= UInt8(102):  # 'a'-'f'
        return Int(byte) - 87
    if byte >= UInt8(65) and byte <= UInt8(70):  # 'A'-'F'
        return Int(byte) - 55
    return 0


struct JsonParser:
    var data: Pointer[UInt8, MutUntrackedOrigin]
    var size: Int
    var pos: Int

    def __init__(out self, data: Pointer[UInt8, MutUntrackedOrigin], size: Int):
        self.data = data
        self.size = size
        self.pos = 0

    def _peek(self) -> UInt8:
        if self.pos < self.size:
            return self.data.unsafe_load[width=1](offset=self.pos)
        return UInt8(0)

    def _advance(mut self) -> UInt8:
        var byte = self._peek()
        self.pos += 1
        return byte

    # Public peek/advance for external streaming consumers (the it-server
    # request-body parser in core/http walks object keys with them).

    def peek(self) -> UInt8:
        return self._peek()

    def advance(mut self) -> UInt8:
        return self._advance()

    def skip(mut self):
        """Advance one byte, discarding it (the value is not needed)."""
        self.pos += 1

    def _skip_ws(mut self):
        while self.pos < self.size:
            var b = self._peek()
            if (
                b == UInt8(32)
                or b == UInt8(9)
                or b == UInt8(10)
                or b == UInt8(13)
            ):
                self.pos += 1
            else:
                break

    def skip_ws(mut self):
        self._skip_ws()

    def expect_byte(mut self, byte: UInt8) raises:
        var got = self._advance()
        if got != byte:
            raise Error("json: unexpected byte")

    def _expect_ascii(mut self, literal: String) raises:
        var bytes = literal.as_bytes()
        for i in range(len(bytes)):
            var got = self._advance()
            if got != bytes[i]:
                raise Error("json: literal mismatch")

    # -- streaming extraction primitives ------------------------------------

    def parse_string(mut self) raises -> String:
        """Parse one JSON string (caller positioned at the opening quote)."""
        var (end, decoded_len) = self._scan_string_end()
        var has_escape = self._string_has_escape(self.pos + 1, end - 1)
        if not has_escape:
            var span = Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=self.data.unsafe_offset(self.pos + 1),
                length=end - self.pos - 2,
            )
            self.pos = end
            return String(unsafe_from_utf8=span)
        var buf = self._alloc_buf(decoded_len)
        var out_idx = 0
        var p = self.pos + 1
        while p < end - 1:
            var b = self.data.unsafe_load[width=1](offset=p)
            if b == UInt8(92):  # '\'
                var esc = self.data.unsafe_load[width=1](offset=p + 1)
                if esc == UInt8(117):  # 'u'
                    var cp = self._read_hex4(p + 2)
                    p += 6
                    if cp >= 0xD800 and cp <= 0xDBFF:
                        # High surrogate: expect a low surrogate next.
                        if (
                            p + 1 < end
                            and self.data.unsafe_load[width=1](offset=p)
                            == UInt8(92)
                            and self.data.unsafe_load[width=1](offset=p + 1)
                            == UInt8(117)
                        ):
                            var low = self._read_hex4(p + 2)
                            cp = (
                                0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00)
                            )
                            p += 6
                    out_idx = utf8_encode(cp, buf, out_idx)
                    continue
                if esc == UInt8(110):  # 'n'
                    buf.unsafe_store(out_idx, UInt8(10))
                elif esc == UInt8(116):  # 't'
                    buf.unsafe_store(out_idx, UInt8(9))
                elif esc == UInt8(114):  # 'r'
                    buf.unsafe_store(out_idx, UInt8(13))
                elif esc == UInt8(98):  # 'b'
                    buf.unsafe_store(out_idx, UInt8(8))
                elif esc == UInt8(102):  # 'f'
                    buf.unsafe_store(out_idx, UInt8(12))
                else:  # '"', '\', '/'
                    buf.unsafe_store(out_idx, esc)
                out_idx += 1
                p += 2
                continue
            buf.unsafe_store(out_idx, b)
            out_idx += 1
            p += 1
        self.pos = end
        var span = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=buf, length=decoded_len
        )
        return String(unsafe_from_utf8=span)

    def parse_int_raw(mut self) raises -> Int:
        """Parse a JSON integer, returning the Int directly (no DOM)."""
        var negative = False
        if self._peek() == UInt8(45):  # '-'
            negative = True
            self.skip()
        var value = 0
        while self._peek() >= UInt8(48) and self._peek() <= UInt8(57):
            value = value * 10 + Int(self._advance()) - 48
        if negative:
            return -value
        return value

    def parse_number(mut self) raises -> JsonScalar:
        var start = self.pos
        var is_float = False
        if self._peek() == UInt8(45):  # '-'
            self.skip()
        while self._peek() >= UInt8(48) and self._peek() <= UInt8(57):
            self.skip()
        if self._peek() == UInt8(46):  # '.'
            is_float = True
            self.skip()
            while self._peek() >= UInt8(48) and self._peek() <= UInt8(57):
                self.skip()
        var b = self._peek()
        if b == UInt8(101) or b == UInt8(69):  # 'e' / 'E'
            is_float = True
            self.skip()
            if self._peek() == UInt8(43) or self._peek() == UInt8(
                45
            ):  # +/- exponent sign
                self.skip()
            while self._peek() >= UInt8(48) and self._peek() <= UInt8(57):
                self.skip()
        var end = self.pos
        var span = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=self.data.unsafe_offset(start), length=end - start
        )
        var text = String(unsafe_from_utf8=span)
        if is_float:
            return JsonScalar.of_float(_atof(text))
        return JsonScalar.of_int(_atoi(text))

    def read_bool_raw(mut self) raises -> Bool:
        var b = self._peek()
        if b == UInt8(116):  # 't'
            self._expect_ascii("true")
            return True
        self._expect_ascii("false")
        return False

    def skip_value(mut self) raises:
        """Advance past one complete JSON value (recursive)."""
        self._skip_ws()
        var b = self._peek()
        if b == UInt8(123):  # '{'
            self.skip()
            self._skip_ws()
            if self._peek() == UInt8(125):  # '}'
                self.skip()
                return
            while True:
                self._skip_ws()
                if self._peek() == UInt8(34):
                    _ = self.parse_string()
                else:
                    raise Error("json: expected object key in skip")
                self._skip_ws()
                self.expect_byte(UInt8(58))  # ':'
                self.skip_value()
                self._skip_ws()
                var c = self._peek()
                if c == UInt8(44):  # ','
                    self.skip()
                    continue
                if c == UInt8(125):  # '}'
                    self.skip()
                    return
                raise Error("json: expected ',' or '}' in skip")
        elif b == UInt8(91):  # '['
            self.skip()
            self._skip_ws()
            if self._peek() == UInt8(93):  # ']'
                self.skip()
                return
            while True:
                self.skip_value()
                self._skip_ws()
                var c = self._peek()
                if c == UInt8(44):  # ','
                    self.skip()
                    continue
                if c == UInt8(93):  # ']'
                    self.skip()
                    return
                raise Error("json: expected ',' or ']' in skip")
        elif b == UInt8(34):  # '"'
            _ = self.parse_string()
        elif b == UInt8(116) or b == UInt8(102) or b == UInt8(110):
            if b == UInt8(116):
                self._expect_ascii("true")
            elif b == UInt8(102):
                self._expect_ascii("false")
            else:
                self._expect_ascii("null")
        elif b == UInt8(45) or (b >= UInt8(48) and b <= UInt8(57)):
            _ = self.parse_number()
        else:
            raise Error("json: unexpected value in skip")

    # -- internals ----------------------------------------------------------

    def _alloc_buf(self, length: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
        from std.memory.alloc import unsafe_alloc

        return unsafe_alloc[UInt8](length)

    def _scan_string_end(mut self) raises -> Tuple[Int, Int]:
        """Return (end_offset, decoded_len) without consuming the string.

        `end_offset` is the offset just past the closing quote; `decoded_len`
        is the UTF-8 byte length after escape processing.  The walk mirrors
        the decode loop in `parse_string` exactly (same hex4/surrogate
        offsets) so the two never disagree about the length.
        """
        var p = self.pos + 1  # past the opening quote
        var decoded = 0
        while p < self.size:
            var b = self.data.unsafe_load[width=1](offset=p)
            if b == UInt8(34):  # '"'
                return (p + 1, decoded)
            if b == UInt8(92):  # '\'
                var esc = self.data.unsafe_load[width=1](offset=p + 1)
                if esc == UInt8(117):  # 'u'
                    var cp = self._read_hex4(p + 2)
                    p += 6
                    if cp >= 0xD800 and cp <= 0xDBFF:
                        if (
                            p + 1 < self.size
                            and self.data.unsafe_load[width=1](offset=p)
                            == UInt8(92)
                            and self.data.unsafe_load[width=1](offset=p + 1)
                            == UInt8(117)
                        ):
                            var low = self._read_hex4(p + 2)
                            cp = (
                                0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00)
                            )
                            p += 6
                    decoded += _utf8_len(cp)
                    continue
                decoded += 1  # other escapes decode to one byte
                p += 2
                continue
            # Raw UTF-8 byte (the file stores non-ASCII directly).
            if b < UInt8(0x80):
                decoded += 1
            else:
                decoded += _utf8_len_raw(self.data, p)
            p += 1
        raise Error("json: unterminated string")

    def _read_hex4(self, p: Int) -> Int:
        var value = 0
        for i in range(4):
            value = value * 16 + hex_value(
                self.data.unsafe_load[width=1](offset=p + i)
            )
        return value

    def _string_has_escape(self, start: Int, end: Int) -> Bool:
        var p = start
        while p < end:
            if self.data.unsafe_load[width=1](offset=p) == UInt8(92):
                return True
            p += 1
        return False


def _utf8_len(cp: Int) -> Int:
    if cp < 0x80:
        return 1
    if cp < 0x800:
        return 2
    if cp < 0x10000:
        return 3
    return 4


def _utf8_len_raw(data: Pointer[UInt8, MutUntrackedOrigin], p: Int) -> Int:
    var b = Int(data.unsafe_load[width=1](offset=p))
    if b >= 0xF0:
        return 4
    if b >= 0x0E0:
        return 3
    if b >= 0xC0:
        return 2
    return 1


def _atoi(text: String) -> Int:
    """Parse a decimal integer (optionally negative)."""
    var value = 0
    var negative = False
    var idx = 0
    var bytes = text.as_bytes()
    if idx < len(bytes) and bytes[idx] == UInt8(45):  # '-'
        negative = True
        idx += 1
    while idx < len(bytes):
        var b = bytes[idx]
        if b >= UInt8(48) and b <= UInt8(57):
            value = value * 10 + Int(b) - 48
        idx += 1
    if negative:
        return -value
    return value


def _atof(text: String) -> Float64:
    """Parse a decimal float (sign, fraction, exponent)."""
    var value = Float64(0)
    var negative = False
    var idx = 0
    var bytes = text.as_bytes()
    if idx < len(bytes) and bytes[idx] == UInt8(45):  # '-'
        negative = True
        idx += 1
    while idx < len(bytes):
        var b = bytes[idx]
        if b >= UInt8(48) and b <= UInt8(57):
            value = value * 10 + Float64(Int(b) - 48)
        else:
            break
        idx += 1
    if idx < len(bytes) and bytes[idx] == UInt8(46):  # '.'
        idx += 1
        var frac = Float64(0.1)
        while idx < len(bytes):
            var b = bytes[idx]
            if b >= UInt8(48) and b <= UInt8(57):
                value += Float64(Int(b) - 48) * frac
                frac *= Float64(0.1)
            else:
                break
            idx += 1
    if idx < len(bytes) and (
        bytes[idx] == UInt8(101) or bytes[idx] == UInt8(69)
    ):
        idx += 1
        var exp_negative = False
        if idx < len(bytes) and bytes[idx] == UInt8(45):  # '-'
            exp_negative = True
            idx += 1
        elif idx < len(bytes) and bytes[idx] == UInt8(43):  # '+'
            idx += 1
        var exp = 0
        while idx < len(bytes):
            var b = bytes[idx]
            if b >= UInt8(48) and b <= UInt8(57):
                exp = exp * 10 + Int(b) - 48
            idx += 1
        var pow10 = Float64(1)
        for _ in range(exp):
            pow10 *= 10
        if exp_negative:
            value = value / pow10
        else:
            value = value * pow10
    if negative:
        return -value
    return value


def parse_json_flat_file(path: String) raises -> Dict[String, JsonScalar]:
    """Parse a flat JSON object file into a `Dict[String, JsonScalar]`.

    Nested values (arrays/objects) are skipped; scalar members are kept.
    """
    from .memory import mmap_file

    var (data, size) = mmap_file(path)
    var parser = JsonParser(data, size)
    var result = Dict[String, JsonScalar]()
    parser.skip_ws()
    parser.expect_byte(UInt8(123))  # '{'
    while True:
        parser.skip_ws()
        if parser._peek() == UInt8(125):  # '}'
            parser.skip()
            break
        var key = parser.parse_string()
        parser.skip_ws()
        parser.expect_byte(UInt8(58))  # ':'
        parser.skip_ws()
        var b = parser._peek()
        if b == UInt8(34):  # '"'
            var value = parser.parse_string()
            result[key] = JsonScalar.of_str(value^)
        elif b == UInt8(116) or b == UInt8(102):  # true / false
            var value = parser.read_bool_raw()
            result[key] = JsonScalar.of_bool(value)
        elif b == UInt8(110):  # null
            parser._expect_ascii("null")
            result[key] = JsonScalar()
        elif b == UInt8(45) or (b >= UInt8(48) and b <= UInt8(57)):
            var value = parser.parse_number()
            result[key] = value^
        else:
            parser.skip_value()
            result[key] = JsonScalar()
        parser.skip_ws()
        if parser._peek() == UInt8(44):  # ','
            parser.skip()
    return result^


def flat_get_int(
    values: Dict[String, JsonScalar], key: String, default: Int
) -> Int:
    var v = values.get(key)
    if v:
        return v.value().as_int()
    return default


def flat_get_float(
    values: Dict[String, JsonScalar], key: String, default: Float64
) -> Float64:
    var v = values.get(key)
    if v:
        return v.value().as_float()
    return default


def flat_get_bool(
    values: Dict[String, JsonScalar], key: String, default: Bool
) -> Bool:
    var v = values.get(key)
    if v:
        return v.value().as_bool()
    return default


def flat_get_str(
    values: Dict[String, JsonScalar], key: String, default: String
) -> String:
    var v = values.get(key)
    if v:
        return v.value().as_str()
    return default
