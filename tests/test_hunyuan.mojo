# tests/test_hunyuan.mojo
#
# M7: Hy-MT2 (hunyuan-dense) load + single-token forward + translation smoke.

from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    TransformerConfig,
    load_config,
    collect_weights,
    ARCH_HUNYUAN,
    arch_name,
)
from src.core.tokenizers import make_tokenizer
from src.core.sampler import Sampler, sample_dynamic, seed_sampler
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple

comptime MODEL_PATH = "Hy-MT2-7B-Q4_K_M.gguf"


def main() raises:
    var ctx = load_gguf(MODEL_PATH)
    var config = load_config(ctx)
    print("arch:", arch_name(config.arch), "layers:", config.n_layers)
    print(
        "hidden:", config.hidden, "ffn:", config.ffn, "heads:", config.n_heads
    )
    print(
        "kv:",
        config.n_kv_heads,
        "head_dim:",
        config.head_dim,
        "vocab:",
        config.vocab,
    )
    print(
        "rope:",
        config.rope_theta,
        "eps:",
        config.norm_eps,
        "bos:",
        config.bos_id,
    )
    check(config.arch == ARCH_HUNYUAN, "arch detection")
    check(config.n_layers == 32, "layer count")
    check(config.hidden == 4096 and config.head_dim == 128, "hidden/head_dim")
    check(config.vocab == 128167, "vocab")

    var weights = collect_weights(ctx)
    var model = TransformerModel(config, ctx^, 512)
    model.weights = weights^

    var tokenizer = make_tokenizer(model.ctx, String(""))
    print(
        "tokenizer:", tokenizer.flavor_name(), "vocab:", tokenizer.vocab_size()
    )

    # single-token forward sanity (position 0)
    var tokens = tokenizer.encode_with_bos(
        "Translate to English: 今天天气很好。"
    )
    print("prompt tokens:", len(tokens))
    var logits = tensor_zeros[DType.float32, 1](
        StaticTuple[Int, 1](config.vocab)
    )
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

    # short translation generation
    seed_sampler(Optional(7))
    var sampler = Sampler(
        temperature=Float32(0.6), top_k=40, top_p=Float32(0.95)
    )
    var generated = List[Int]()
    for _ in range(64):
        var next_token = sample_dynamic[DType.float32](logits, sampler, tokens)
        if next_token == tokenizer.eos_id():
            break
        generated.append(next_token)
        tokens.append(next_token)
        logits = model.forward(next_token, len(tokens) - 1)
    var output = tokenizer.decode(generated)
    print("generated:", output)
    print("test_hunyuan OK")


def check(cond: Bool, name: String):
    if not cond:
        print("FAIL:", name)
        abort()


def abort():
    from std.os.os import abort as _abort

    _abort()
