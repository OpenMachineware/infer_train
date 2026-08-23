# main.mojo
#
# M3 end-to-end: load DeepSeek-R1-Distill-Qwen-1.5B (Q5_K_M), tokenize with
# the real Qwen2 BPE tokenizer, and generate a completion token by token.

from src.runtime.inference import load_model, generate


def main() raises:
    var model = load_model(
        "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"
    )
    var cfg = model.transformer.config
    print("architecture config:")
    print("  layers:", cfg.n_layers)
    print("  hidden:", cfg.hidden)
    print("  ffn:", cfg.ffn)
    print("  heads:", cfg.n_heads, "kv_heads:", cfg.n_kv_heads)
    print("  head_dim:", cfg.head_dim)
    print("  vocab:", cfg.vocab)
    print("  rope_theta:", cfg.rope_theta)
    print("  norm_eps:", cfg.norm_eps)
    print("  bos:", cfg.bos_id, "eos:", cfg.eos_id)
    print("  weights:", len(model.transformer.weights))
    print("  graph nodes:", len(model.graph.nodes))
    print("  kv cache:", model.transformer.cache.capacity(), "positions")

    # Qwen2 chat prompt for DeepSeek-R1: BOS + User turn + Assistant think.
    var prompt = "<｜User｜>What is 1+1?<｜Assistant｜><think>\n"
    print("prompt tokens:", model.tokenizer.encode_with_bos(prompt))
    print("generating...")
    var output = generate(
        model,
        prompt,
        max_tokens=120,
        temperature=0.6,
        top_p=0.95,
        top_k=40,
        verbose=True,
        seed=7,
    )
    print("generated:", output)
