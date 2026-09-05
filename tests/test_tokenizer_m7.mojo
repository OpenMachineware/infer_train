# tests/test_tokenizer_m7.mojo
#
# M7 tokenizer abstraction tests:
#   * flavor auto-selection from GGUF metadata (qwen / llama / hunyuan);
#   * hunyuan (Hy-MT2) and qwen35 encode parity with llama.cpp
#     (`llama-tokenize --stdin --ids` reference outputs);
#   * the custom-tokenizer registry (`register_tokenizer` /
#     `make_tokenizer` override);
#   * the trait contract through a generic helper.

from src.core.gguf_loader import load_gguf
from src.core.tokenizer import Tokenizer, TokenizerFlavor
from src.core.tokenizers import (
    make_tokenizer,
    TokenizerSpec,
    TokenizerRegistry,
    register_tokenizer,
    BpeTokenizer,
)
from src.core.tokenizers.bpe_engine import (
    FLAVOR_QWEN,
    FLAVOR_HUNYUAN,
    FLAVOR_CUSTOM,
)

comptime MODEL_1_5B = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"
comptime MODEL_HY = "Hy-MT2-7B-Q4_K_M.gguf"
# The qwen35 tokenizer reference.  Was Qwen3.8-27B-UD-Q5_K_M.gguf; that file
# is not present on this machine, so the Qwen3.6-35B-A3B GGUF stands in -
# its tokenizer metadata (pre=qwen35, vocab 248320, bos/eos, merges) is
# identical for every check below.
comptime MODEL_QWEN35 = (
    "Qwen3.6-35B-A3B-DSV4Pro-Distill-MTP-Q5_K_M-imatrix.gguf"
)


def main() raises:
    # ---- 1.5B (qwen flavor, tokenizer.json path) --------------------------
    var ctx_qwen = load_gguf(MODEL_1_5B)
    var tok = make_tokenizer(ctx_qwen, "tests/data/tokenizer.json")
    check(tok.flavor_name() == "qwen", "1.5B flavor == qwen")
    check(tok.vocab_size() == 151936, "1.5B vocab_size")
    check(tok.bos_id() == 151646, "1.5B bos")
    check(tok.eos_id() == 151643, "1.5B eos")
    check_enc(tok, "Hello", [9707])
    var with_bos = tok.encode_with_bos("Hello")
    check(
        len(with_bos) == 2 and with_bos[0] == 151646 and with_bos[1] == 9707,
        "encode_with_bos",
    )

    # ---- Hy-MT2 (hunyuan flavor, GGUF-only: no tokenizer.json) ------------
    var ctx_hy = load_gguf(MODEL_HY)
    var hy = make_tokenizer(ctx_hy, String(""))
    check(hy.flavor_name() == "hunyuan", "Hy-MT2 flavor == hunyuan")
    check(hy.vocab_size() == 128167, "Hy-MT2 vocab_size")
    check(hy.bos_id() == 127958, "Hy-MT2 bos")
    check(hy.eos_id() == 3, "Hy-MT2 eos")
    check(len(hy._added) > 0, "Hy-MT2 derived added tokens")
    # llama.cpp reference (`llama-tokenize --stdin --ids`):
    # [28573, 311, 6498, 25, 220, 104944, 105242, 102722, 1811]
    check_enc(
        hy,
        "Translate to English: 今天天气很好。",
        [28573, 311, 6498, 25, 220, 104944, 105242, 102722, 1811],
    )
    check_roundtrip(hy, "Translate to English: 今天天气很好。")
    check_roundtrip(hy, "Hello, world! 123")

    # ---- qwen35 (qwen35 -> qwen flavor, GGUF-only) -------------------------
    var ctx_35 = load_gguf(MODEL_QWEN35)
    var q35 = make_tokenizer(ctx_35, String(""))
    check(q35.flavor_name() == "qwen", "qwen35 flavor == qwen")
    check(q35.vocab_size() == 248320, "qwen35 vocab_size")
    check(q35.bos_id() == 248044, "qwen35 bos")
    check(q35.eos_id() == 248046, "qwen35 eos")
    # llama.cpp reference: [9419, 1814]
    check_enc(q35, "Hello world", [9419, 1814])

    # ---- custom registry ---------------------------------------------------
    var registry = TokenizerRegistry()
    var spec = TokenizerSpec("qwen2", TokenizerFlavor.Custom, 42, 43)
    register_tokenizer(registry, spec^)
    check(registry.size() == 1, "registry size")
    var custom = make_tokenizer(ctx_qwen, String(""), Optional(registry^))
    check(custom.flavor_name() == "custom", "registry override flavor")
    check(
        custom.bos_id() == 42 and custom.eos_id() == 43, "registry override ids"
    )

    # ---- generic engine + trait conformance --------------------------------
    # One BPE engine for every BPE family: the flavor is data (tag +
    # added-token table), auto-selected from the GGUF metadata.
    var qwen_tok = BpeTokenizer.load("tests/data/tokenizer.json", ctx_qwen)
    var hy_tok = make_tokenizer(ctx_hy, String(""))
    print_tok_info[BpeTokenizer](qwen_tok)
    print_tok_info[BpeTokenizer](hy_tok)
    check(qwen_tok.vocab_size() == 151936, "qwen engine vocab")
    check(hy_tok.bos_id() == 127958, "hunyuan engine bos")
    check(hy_tok.vocab_size() == 128167, "hunyuan engine vocab")

    print("test_tokenizer_m7 OK")


def print_tok_info[T: Tokenizer](t: T):
    var n = t.vocab_size()
    if n <= 0:
        print("FAIL: empty vocab")
        abort()


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def check_enc(tok: BpeTokenizer, text: String, expected: List[Int]):
    var tokens = tok.encode(text)
    if len(tokens) != len(expected):
        print(
            "FAIL len:",
            text,
            "actual:",
            len(tokens),
            "expected:",
            len(expected),
        )
        for t in tokens:
            print("  got", t)
        abort()
    for i in range(len(expected)):
        if tokens[i] != expected[i]:
            print(
                "FAIL:",
                text,
                "index",
                i,
                "actual:",
                tokens[i],
                "expected:",
                expected[i],
            )
            abort()


def check_roundtrip(tok: BpeTokenizer, text: String):
    var tokens = tok.encode(text)
    var decoded = tok.decode(tokens)
    if decoded != text:
        print("FAIL roundtrip:", text, "->", decoded)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
