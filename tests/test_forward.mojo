# tests/test_forward.mojo
#
# Forward validation: feed the prompt "1+1=" (BOS + tokens) through the real
# transformer and print the per-step top-5 logits.  Compare against the
# llama.cpp reference (reference_logits_prompt.npy / llama-cli logits file).

from src.core.gguf_loader import load_gguf
from src.core.transformer import (
    TransformerModel,
    load_config,
    collect_weights,
    DEFAULT_KV_CACHE_LEN,
)
from src.core.tokenizers.bpe_engine import BpeTokenizer


def main() raises:
    var ctx = load_gguf("DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf")
    var config = load_config(ctx)
    var weights = collect_weights(ctx)
    var model = TransformerModel(config, ctx^, DEFAULT_KV_CACHE_LEN)
    model.weights = weights^
    print("weights loaded")

    var tokenizer = BpeTokenizer.load("tokenizer.json", model.ctx)
    var prompt_tokens = tokenizer.encode_with_bos("1+1=")
    print("prompt tokens:", len(prompt_tokens))

    for i in range(len(prompt_tokens)):
        var logits = model.forward(prompt_tokens[i], i)
        print("step", i, "token", prompt_tokens[i])
        # top-5
        var top5 = List[Int]()
        var top5v = List[Float32]()
        for k in range(5):
            var best = -1
            var best_v = Float32(-3.0e38)
            for j in range(logits.shape()[0]):
                if j in top5:
                    continue
                var v = Float32(logits.get(j))
                if v > best_v:
                    best_v = v
                    best = j
            top5.append(best)
            top5v.append(best_v)
        for k in range(5):
            print("   ", top5[k], top5v[k])
    print("test_forward OK")
