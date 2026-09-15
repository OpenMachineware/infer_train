# Decode performance breakdown benchmark
from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
    ARCH_QWEN3,
    arch_name,
)
from src.core.tokenizers import make_tokenizer
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
    
    var tokenizer = make_tokenizer(model.ctx, String(""))
    
    # Prefill first
    var prompt = "Translate to English: 今天天气很好，我想出去散步。"
    var tokens = tokenizer.encode_with_bos(prompt)
    print("Prefill tokens:", len(tokens))
    var logits = model.forward_batch(tokens, 0, 64)
    
    # Now measure decode with timing breakdown
    var n_decode = 16
    var total_ms = 0
    
    for i in range(n_decode):
        var next_token = 100 + i  # dummy token
        var start = now_ns()
        logits = model.forward(next_token, len(tokens))
        var end = now_ns()
        total_ms += (end - start) // 1_000_000
        tokens.append(next_token)
    
    print("\n=== Decode Breakdown (context =", len(tokens) - n_decode, "tokens) ===")
    print("Total for", n_decode, "tokens:", total_ms, "ms")
    print("Per token:", Float64(total_ms) / Float64(n_decode), "ms")
    print("Speed:", Float64(n_decode) * 1000.0 / Float64(total_ms), "t/s")
    
    # Test with longer context
    print("\n=== Testing with longer context ===")
    var long_prompt = "The quick brown fox jumps over the lazy dog. " * 10
    var long_tokens = tokenizer.encode_with_bos(long_prompt)
    logits = model.forward_batch(long_tokens, 0, 64)
    print("Long context tokens:", len(long_tokens))
    
    var long_decode = 8
    var long_ms = 0
    for i in range(long_decode):
        var next_token = 200 + i
        var start = now_ns()
        logits = model.forward(next_token, len(long_tokens))
        var end = now_ns()
        long_ms += (end - start) // 1_000_000
        long_tokens.append(next_token)
    
    print("Decode with", len(long_tokens) - long_decode, "context tokens:")
    print("  Total for", long_decode, "tokens:", long_ms, "ms")
    print("  Per token:", Float64(long_ms) / Float64(long_decode), "ms")
    print("  Speed:", Float64(long_decode) * 1000.0 / Float64(long_ms), "t/s")
    
    # Detailed timing for one layer
    print("\n=== Per-layer timing (Layer 0) ===")
    model.cache.layers[0].filled = 0  # Reset cache
    
    var dummy_scale = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](1))
    
    var x = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](1, config.hidden))
    for d in range(config.hidden):
        x.set(d, Scalar[DType.float16](Float16(0.1)))
    
    var lw = model.layer_view(0)
    
    # QKV projections
    var t0 = now_ns()
    var q = lw.q_w.proj(x, dummy_scale)
    var k = lw.k_w.proj(x, dummy_scale)
    var v = lw.v_w.proj(x, dummy_scale)
    var t1 = now_ns()
    print("QKV proj:", (t1 - t0) // 1000, "us")
    
    # Output projection timing
    var dummy_out = tensor_zeros[DType.float16, 2](
        StaticTuple[Int, 2](1, config.n_heads * config.head_dim)
    )
    var t2 = now_ns()
    var o = lw.o_w.proj(dummy_out, dummy_scale)
    var t3 = now_ns()
    print("O proj:", (t3 - t2) // 1000, "us")
    
    # FFN timing
    var dummy_normed = tensor_zeros[DType.float16, 2](
        StaticTuple[Int, 2](1, config.hidden)
    )
    var t4 = now_ns()
    var g = lw.gate_w.proj(dummy_normed, dummy_scale)
    var u = lw.up_w.proj(dummy_normed, dummy_scale)
    var t5 = now_ns()
    print("FFN gate+up:", (t5 - t4) // 1000, "us")
    
    var dummy_h = tensor_zeros[DType.float16, 2](
        StaticTuple[Int, 2](1, config.ffn)
    )
    var t6 = now_ns()
    var d = lw.down_w.proj(dummy_h, dummy_scale)
    var t7 = now_ns()
    print("FFN down:", (t7 - t6) // 1000, "us")
    
    print("\n=== Matmul counts per token ===")
    print("Per layer:")
    print("  QKV: 3 matmuls (Q, K, V)")
    print("  O: 1 matmul")
    print("  FFN: 3 matmuls (gate, up, down)")
    print("  Total: 7 matmuls per layer")
    print("Full model:", config.n_layers, "layers x 7 = ", config.n_layers * 7, "matmuls per token")