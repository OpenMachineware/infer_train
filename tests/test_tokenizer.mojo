# tests/test_tokenizer.mojo
#
# Qwen2 BPE tokenizer validation against llama.cpp reference output
# (tests/reference/reference_dump.py + reference_tokenizer.json).

from src.core.gguf_loader import load_gguf
from src.core.tokenizers.bpe_engine import BpeTokenizer

comptime MODEL_PATH = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"


def main() raises:
    var ctx = load_gguf(MODEL_PATH)
    var tokenizer = BpeTokenizer.load("tests/data/tokenizer.json", ctx)
    print("vocab:", len(tokenizer._vocab))
    print("merges:", len(tokenizer._merges))
    print("decode table:", len(tokenizer._decode_table))
    print("added:", len(tokenizer._added))
    print("bos:", tokenizer.bos_id(), "eos:", tokenizer.eos_id())

    # plain-text cases: must match llama.cpp exactly
    check_enc(tokenizer, "Hello", [9707])
    check_enc(tokenizer, "Hello world", [9707, 1879])
    check_enc(tokenizer, "Hello, world! 123", [9707, 11, 1879, 0, 220, 16, 17, 18])
    check_enc(tokenizer, "1+1=", [16, 10, 16, 28])
    check_enc(tokenizer, "2+2=?", [17, 10, 17, 19884])
    check_enc(tokenizer, "What is the capital of France?", [3838, 374, 279, 6722, 315, 9625, 30])
    check_enc(tokenizer, "The capital of France is", [785, 6722, 315, 9625, 374])
    check_enc(tokenizer, "   leading spaces", [256, 6388, 12621])
    check_enc(tokenizer, "new\nline", [931, 198, 1056])
    check_enc(tokenizer, "café", [924, 58858])
    check_enc(tokenizer, "你好世界", [108386, 99489])
    check_enc(
        tokenizer,
        "It's a test. It's 'ok'. 100%",
        [2132, 594, 264, 1273, 13, 1084, 594, 364, 562, 4427, 220, 16, 15, 15, 4],
    )

    # added-token cases: HF-style ids (llama.cpp differs on some of these)
    check_enc(tokenizer, "<think>", [151648])
    check_enc(tokenizer, "</think>", [151649])
    check_enc(tokenizer, "<｜User｜>hi<｜Assistant｜>", [151644, 6023, 151645])
    check_enc(tokenizer, "<｜begin▁of▁sentence｜>", [151646])
    check_enc(tokenizer, "<｜end▁of▁sentence｜>", [151643])

    # bos handling
    var with_bos = tokenizer.encode_with_bos("Hello")
    if len(with_bos) != 2 or with_bos[0] != 151646 or with_bos[1] != 9707:
        print("FAIL encode_with_bos")
        abort()

    # decode round-trips
    check_roundtrip(tokenizer, "Hello world")
    check_roundtrip(tokenizer, "1+1=")
    check_roundtrip(tokenizer, "   leading spaces")
    check_roundtrip(tokenizer, "new\nline")
    check_roundtrip(tokenizer, "café")
    check_roundtrip(tokenizer, "你好世界")
    check_roundtrip(tokenizer, "<think>")
    check_roundtrip(tokenizer, "<｜User｜>hi<｜Assistant｜>")

    print("test_tokenizer OK")


def check_enc(
    tokenizer: BpeTokenizer, text: String, expected: List[Int]
):
    var tokens = tokenizer.encode(text)
    if len(tokens) != len(expected):
        print("FAIL len:", text, "actual:", len(tokens), "expected:", len(expected))
        for t in tokens:
            print("  got", t)
        abort()
    for i in range(len(expected)):
        if tokens[i] != expected[i]:
            print("FAIL:", text, "index", i, "actual:", tokens[i], "expected:", expected[i])
            abort()


def check_roundtrip(tokenizer: BpeTokenizer, text: String):
    var tokens = tokenizer.encode(text)
    var decoded = tokenizer.decode(tokens)
    if decoded != text:
        print("FAIL roundtrip:", text, "->", decoded)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
