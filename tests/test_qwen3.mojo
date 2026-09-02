# tests/test_qwen3.mojo
#
# Qwen3 (arch `qwen3`) load + config + single-token forward + generation
# smoke test for Qwen3-0.6B-UD-Q4_K_XL (Q4_K/Q5_K/Q6_K/IQ4_XS mix, tied
# embeddings, per-head Q/K RMSNorm before RoPE).
#
# Skips (SKIP) if the model file is not present.

from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
    ARCH_QWEN3,
    arch_name,
)
from src.core.tokenizers import make_tokenizer
from src.core.sampler import Sampler, sample_dynamic, seed_sampler
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple

comptime MODEL_PATH = "Qwen3-0.6B-UD-Q4_K_XL.gguf"


def _file_exists(path: String) -> Bool:
    try:
        var f = FileHandle(path, "r")
        f.close()
        return True
    except:
        return False


def main() raises:
    if not _file_exists(MODEL_PATH):
        print("SKIP: " + MODEL_PATH + " not present")
        return

    var ctx = load_gguf(MODEL_PATH)
    var config = load_config(ctx)
    print("arch:", arch_name(config.arch), "layers:", config.n_layers)
    print("hidden:", config.hidden, "ffn:", config.ffn, "heads:", config.n_heads)
    print("kv:", config.n_kv_heads, "head_dim:", config.head_dim, "vocab:", config.vocab)
    print("rope:", config.rope_theta, "eps:", config.norm_eps)
    check(config.arch == ARCH_QWEN3, "arch detection")
    check(config.n_layers == 28, "layer count")
    check(config.hidden == 1024 and config.ffn == 3072, "hidden/ffn")
    check(config.n_heads == 16 and config.n_kv_heads == 8, "heads")
    # head_dim must come from the metadata key_length (128), NOT from
    # hidden/n_heads (64) - the Qwen3-0.6B special case.
    check(config.head_dim == 128, "head_dim from key_length")
    check(config.vocab == 151936, "vocab")
    check(config.rope_theta > Float32(999999.0), "rope theta 1e6")

    var weights = collect_weights(ctx)
    var model = TransformerModel(config, ctx^, 512)
    model.weights = weights^

    var tokenizer = make_tokenizer(model.ctx, String(""))
    print("tokenizer:", tokenizer.flavor_name(), "vocab:", tokenizer.vocab_size())

    # single-token forward sanity (prefill a few prompt tokens)
    var tokens = tokenizer.encode_with_bos("What is 1+1?")
    print("prompt tokens:", len(tokens))
    var logits = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](config.vocab))
    for i in range(len(tokens)):
        logits = model.forward(tokens[i], i)
    var mx = Float32(-3.0e38)
    var arg = 0
    for i in range(config.vocab):
        var v = Float32(logits.get(i))
        if v > mx:
            mx = v
            arg = i
    var one = List[Int]()
    one.append(arg)
    print("argmax:", arg, "value:", mx, "decoded:", tokenizer.decode(one))
    check(mx > Float32(-1.0e6) and mx < Float32(1.0e6), "logits finite")

    # short generation
    seed_sampler(Optional(7))
    var sampler = Sampler(temperature=Float32(0.6), top_k=40, top_p=Float32(0.95))
    var generated = List[Int]()
    for _ in range(48):
        var next_token = sample_dynamic[DType.float32](logits, sampler, tokens)
        if next_token == tokenizer.eos_id():
            break
        generated.append(next_token)
        tokens.append(next_token)
        logits = model.forward(next_token, len(tokens) - 1)
    var output = tokenizer.decode(generated)
    print("generated:", output)
    check(len(generated) > 0, "generated tokens")
    print("test_qwen3 OK")


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
