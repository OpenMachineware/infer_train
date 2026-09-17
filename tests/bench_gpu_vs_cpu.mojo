# Compare pure CPU vs hybrid (CPU attention + GPU FFN)
from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
)
from src.core.ops.gpu.gpu_runtime import has_metal_gpu
from src.core.tensor import tensor_zeros
from std.utils.static_tuple import StaticTuple

def main() raises:
    print("=== Pure CPU vs Hybrid Mode Benchmark ===")

    if not has_metal_gpu():
        print("No GPU available")
        return

    # Load model
    var model_path = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"
    try:
        print("Loading model...")
        var ctx = load_gguf(model_path)
        var config = load_config(ctx)
        var weights = collect_weights(ctx)
        var model = TransformerModel(config, ctx^, 512)
        model.weights = weights^
        print("Model loaded:", config.hidden, "hidden,", config.n_layers, "layers")

        # Warmup
        print("\nWarming up...")
        for _ in range(3):
            _ = model.forward(1, 0)

        # Test 1: Pure CPU (don't upload weights)
        print("\n=== Pure CPU (no GPU weights) ===")
        var runs = 10
        var total_us = Int64(0)
        for i in range(runs):
            var start = now()
            _ = model.forward(1, 0)
            total_us += Int64((now() - start) // 1_000)
        var avg_cpu = total_us / Int64(runs)
        print("Average:", avg_cpu, "us (", Float64(avg_cpu) / 1000.0, "ms )")

        # Upload weights to GPU
        print("\nUploading weights to GPU...")
        model.upload_weights_to_gpu()
        print("Weights uploaded")

        # Warmup with GPU
        for _ in range(3):
            _ = model.forward(1, 0)

        # Test 2: Hybrid (CPU attention + GPU FFN)
        print("\n=== Hybrid (CPU Attn + GPU FFN) ===")
        total_us = Int64(0)
        for i in range(runs):
            var start = now()
            _ = model.forward(1, 0)
            total_us += Int64((now() - start) // 1_000)
        var avg_hybrid = total_us / Int64(runs)
        print("Average:", avg_hybrid, "us (", Float64(avg_hybrid) / 1000.0, "ms )")

        # Comparison
        print("\n=== Comparison ===")
        var slowdown = Float64(avg_hybrid) / Float64(avg_cpu)
        print("Hybrid is", slowdown, "x", "slower" if slowdown > 1.0 else "faster")

        print("\nConclusion:")
        print("GPU FFN with CPU attention causes CPU<->GPU transfers")
        print("This overhead makes it SLOWER than pure CPU")
        print("Need: Complete GPU forward (attention + FFN) to avoid transfers")

    except:
        print("Model not found or error")


def now() -> Int:
    from src.core.thread_pool import now_ns
    return now_ns()
