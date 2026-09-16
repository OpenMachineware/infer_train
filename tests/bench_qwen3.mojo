# Performance benchmark for Qwen3-0.6B model
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
from src.core.thread_pool import now_ns
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
    print("Model:", arch_name(config.arch))
    print("Layers:", config.n_layers, "Hidden:", config.hidden)
    print("FFN:", config.ffn, "Heads:", config.n_heads)

    var weights = collect_weights(ctx)
    var model = TransformerModel(config, ctx^, 512)
    model.weights = weights^
    model.load_weights_quant()

    # Upload all weights to GPU for persistent caching
    model.upload_weights_to_gpu()
    print("Weights uploaded to GPU")

    var tokenizer = make_tokenizer(model.ctx, String(""))

    # Warmup run
    var warmup_tokens = tokenizer.encode_with_bos("Hello")
    var logits = tensor_zeros[DType.float32, 1](StaticTuple[Int, 1](config.vocab))
    for i in range(len(warmup_tokens)):
        logits = model.forward(warmup_tokens[i], i)
    print("Warmup done")

    # Prefill benchmark - use batch path
    var prompt = "Translate to English: 今天天气很好，我想出去散步。这是一个很好的机会去享受阳光和新鲜空气。人工智能正在改变我们的生活方式。" * 3
    var tokens = tokenizer.encode_with_bos(prompt)
    print("Prompt tokens:", len(tokens))

    # Use batch prefill path (faster)
    var prefill_start = now_ns()
    logits = model.forward_batch(tokens, 0, 64)
    var prefill_end = now_ns()
    var prefill_ms = (prefill_end - prefill_start) // 1_000_000
    var prefill_tps = Float64(len(tokens)) * 1000.0 / Float64(prefill_ms)
    print("Prefill:", len(tokens), "tokens in", prefill_ms, "ms =", prefill_tps, "t/s")

    # Decode benchmark
    seed_sampler(Optional(42))
    var sampler = Sampler(temperature=Float32(0.6), top_k=40, top_p=Float32(0.95))

    var decode_tokens = 128
    var generated = List[Int]()

    var decode_start = now_ns()
    for _ in range(decode_tokens):
        var next_token = sample_dynamic[DType.float32](logits, sampler, tokens)
        if next_token == tokenizer.eos_id():
            break
        generated.append(next_token)
        tokens.append(next_token)
        logits = model.forward(next_token, len(tokens) - 1)
    var decode_end = now_ns()

    var decode_ms = (decode_end - decode_start) // 1_000_000
    var decode_tps = Float64(len(generated)) * 1000.0 / Float64(decode_ms)
    print("Decode:", len(generated), "tokens in", decode_ms, "ms =", decode_tps, "t/s")

    # Summary
    print("\n=== Performance Summary ===")
    print("Prefill speed:", prefill_tps, "tokens/s")
    print("Decode speed:", decode_tps, "tokens/s")

    # Comparison with llama.cpp
    print("\n=== Comparison with llama.cpp (4 threads) ===")
    print("llama.cpp prefill (pp512): 570.15 t/s")
    print("llama.cpp decode (tg128): 140.84 t/s")
    print("Our prefill:", prefill_tps, "t/s")
    print("Our decode:", decode_tps, "t/s")
    var prefill_ratio = prefill_tps / 570.15
    var decode_ratio = decode_tps / 140.84
    print("Prefill ratio:", prefill_ratio, "x")
    print("Decode ratio:", decode_ratio, "x")
