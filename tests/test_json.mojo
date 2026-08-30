# tests/test_json.mojo
#
# Unit test for the hand-rolled JSON parser (src/core/json.mojo).

from src.core.json import (
    JsonParser,
    JsonScalar,
    parse_json_flat_file,
    flat_get_int,
    flat_get_float,
    flat_get_bool,
    flat_get_str,
)
from src.core.memory import mmap_file
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin


def _bytes(text: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    var span = text.as_bytes()
    var p = unsafe_alloc[UInt8](len(span))
    for i in range(len(span)):
        p.unsafe_store(i, span[i])
    return p


def parse(doc: String) raises -> JsonScalar:
    var parser = JsonParser(_bytes(doc), doc.byte_length())
    parser.skip_ws()
    return parser.parse_number()


def parse_str(doc: String) raises -> String:
    var parser = JsonParser(_bytes(doc), doc.byte_length())
    parser.skip_ws()
    return parser.parse_string()


def main() raises:
    # scalar parsing
    print("step: scalar int")
    assert_equal(parse("42").as_int(), 42, "int")
    assert_equal(parse("-7").as_int(), -7, "neg int")
    assert_equal(parse("151936").as_int(), 151936, "big int")
    var f = parse("3.5").as_float()
    assert_true(f > 3.4 and f < 3.6, "float")
    var e = parse("1e-6").as_float()
    assert_true(e > 0.9e-6 and e < 1.1e-6, "exponent")
    var big = parse("1.5e10").as_float()
    assert_true(big > 1.4e10 and big < 1.6e10, "big exponent")

    print("step: string escapes")
    # string escapes: "a\n\tAé😀"
    var s = parse_str('"a\\n\\t\\u0041\\u00e9\\ud83d\\ude00"')
    var expect = String("a\n\tAé😀")
    assert_true(s == expect, "escape mismatch: '" + s + "'")
    var raw = parse_str('"hello world"')
    assert_true(raw == "hello world", "raw string")

    print("step: bool literals")
    # bool literals
    var parser2 = JsonParser(_bytes(String("true false null")), 12)
    assert_true(parser2.read_bool_raw(), "true")
    parser2.skip_ws()
    assert_true(not parser2.read_bool_raw(), "false")

    print("step: flat config")
    # flat config file
    var cfg = parse_json_flat_file("tests/data/config.json")
    assert_equal(flat_get_int(cfg, "hidden_size", 0), 1536, "hidden_size")
    assert_equal(flat_get_int(cfg, "num_hidden_layers", 0), 28, "layers")
    assert_equal(
        flat_get_int(cfg, "num_attention_heads", 0), 12, "n_heads"
    )
    assert_equal(
        flat_get_int(cfg, "intermediate_size", 0), 8960, "ffn"
    )
    assert_equal(flat_get_int(cfg, "vocab_size", 0), 151936, "vocab")
    var eps = flat_get_float(cfg, "rms_norm_eps", 0)
    assert_true(eps > 0.9e-6 and eps < 1.1e-6, "rms_norm_eps")
    var theta = flat_get_float(cfg, "rope_theta", 0)
    assert_true(theta > 9999 and theta < 10001, "rope_theta")
    assert_true(
        flat_get_str(cfg, "model_type", "") == "qwen2", "model_type"
    )

    print("step: streaming walk")
    # streaming walk over the real 7 MB tokenizer.json
    var (data, size) = mmap_file("tests/data/tokenizer.json")
    var tok = JsonParser(data, size)
    tok.expect_byte(UInt8(123))  # '{'
    var vocab_count = 0
    var merge_count = 0
    while True:
        tok.skip_ws()
        if tok._peek() == UInt8(125):  # '}'
            _ = tok._advance()
            break
        var key = tok.parse_string()
        tok.skip_ws()
        tok.expect_byte(UInt8(58))  # ':'
        tok.skip_ws()
        if key == "model":
            _ = tok._advance()
            while True:
                tok.skip_ws()
                if tok._peek() == UInt8(125):  # '}'
                    _ = tok._advance()
                    break
                var mkey = tok.parse_string()
                tok.skip_ws()
                tok.expect_byte(UInt8(58))  # ':'
                tok.skip_ws()
                if mkey == "vocab":
                    _ = tok._advance()
                    tok.skip_ws()
                    if tok._peek() != UInt8(125):  # '}'
                        while True:
                            tok.skip_ws()
                            _ = tok.parse_string()
                            tok.skip_ws()
                            tok.expect_byte(UInt8(58))  # ':'
                            tok.skip_ws()
                            _ = tok.parse_int_raw()
                            vocab_count += 1
                            tok.skip_ws()
                            var c = tok._peek()
                            if c == UInt8(44):  # ','
                                _ = tok._advance()
                                continue
                            _ = tok._advance()  # '}'
                            break
                elif mkey == "merges":
                    _ = tok._advance()  # '['
                    tok.skip_ws()
                    if tok._peek() != UInt8(93):  # ']'
                        while True:
                            tok.skip_ws()
                            _ = tok.parse_string()
                            merge_count += 1
                            tok.skip_ws()
                            var c = tok._peek()
                            if c == UInt8(44):  # ','
                                _ = tok._advance()
                                continue
                            _ = tok._advance()  # ']'
                            break
                else:
                    tok.skip_value()
                tok.skip_ws()
                if tok._peek() == UInt8(44):  # ','
                    _ = tok._advance()
                    continue
        else:
            tok.skip_value()
        tok.skip_ws()
        if tok._peek() == UInt8(44):  # ','
            _ = tok._advance()
    print("streamed vocab entries =", vocab_count)
    print("streamed merges =", merge_count)
    assert_equal(vocab_count, 151643, "vocab count")
    assert_equal(merge_count, 151387, "merges count")
    print("test_json OK")


def assert_equal(actual: Int, expected: Int, label: String):
    if actual != expected:
        print("FAIL", label, "actual:", actual, "expected:", expected)
        abort()


def assert_true(condition: Bool, label: String):
    if not condition:
        print("FAIL", label)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
