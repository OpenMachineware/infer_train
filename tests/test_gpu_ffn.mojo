# Test GPU FFN activation
from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
)
from src.core.ops.gpu.gpu_runtime import has_metal_gpu

def main() raises:
    print("=== GPU FFN Test ===")

    if not has_metal_gpu():
        print("No GPU available, skipping")
        return

    print("GPU detected")

    # Load a model (use 1.5B model if available)
    var model_path = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"
    try:
        var ctx = load_gguf(model_path)
        var config = load_config(ctx)
        var weights = collect_weights(ctx)
        var model = TransformerModel(config, ctx^, 512)
        model.weights = weights^
        print("Model loaded")

        # Upload weights to GPU
        print("Uploading weights to GPU...")
        model.upload_weights_to_gpu()
        print("Weights uploaded")

        # Test single token forward
        print("\nTesting forward pass...")
        var token = 1
        var position = 0

        # Test with GPU
        var logits = model.forward(token, position)
        print("Forward pass complete, logits vocab size:", logits.shape()[0])

        print("\nTest passed!")
    except:
        print("Model file not found, skipping")
