# Benchmark GPU FFN end-to-end
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
    print("=== GPU FFN End-to-End Benchmark ===")

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

        # Upload weights to GPU
        print("\nUploading weights to GPU...")
        var start = now()
        model.upload_weights_to_gpu()
        var upload_time = Int64((now() - start) // 1_000_000)
        print("Upload time:", upload_time, "ms")

        # Warmup
        print("\nWarming up...")
        for _ in range(3):
            _ = model.forward(1, 0)

        # Benchmark CPU-only forward
        print("\n=== CPU Forward ===")
        var runs = 10
        var total_us = Int64(0)
        for i in range(runs):
            start = now()
            _ = model.forward(1, 0)
            total_us += Int64((now() - start) // 1_000)

        var avg_us = total_us / Int64(runs)
        print("Average:", avg_us, "us")

        # Check if GPU was used
        print("\nNote: Check if 'use_gpu=True' is actually used in FFN")
        print("Current status: FFN has use_gpu=True, but attention is CPU")
        print("This may cause CPU<->GPU transfers, potentially slower than pure CPU")

    except:
        print("Model not found or error")


def now() -> Int:
    from src.core.thread_pool import now_ns
    return now_ns()
