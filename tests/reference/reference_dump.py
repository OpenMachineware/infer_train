#!/usr/bin/env python3
"""Generate reference data (tokenizer + logits) from llama-cpp-python.

Serves as ground truth for validating the Mojo M3 implementation.
"""
import json
import sys
import numpy as np

from llama_cpp import Llama

MODEL = "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf"

llm = Llama(
    model_path=MODEL,
    n_ctx=512,
    n_threads=10,
    n_batch=64,
    verbose=False,
)

# --- tokenizer round-trip tests -------------------------------------------
test_strings = [
    "Hello",
    "Hello world",
    "Hello, world! 123",
    "1+1=",
    "2+2=?",
    "What is the capital of France?",
    "The capital of France is",
    "<think>",
    "<｜User｜>hi<｜Assistant｜>",
    "   leading spaces",
    "new\nline",
    "café",
    "你好世界",
    "It's a test. It's 'ok'. 100%",
]

tokenizer_results = []
for text in test_strings:
    toks = llm.tokenize(text.encode("utf-8"), add_bos=False)
    detok = llm.detokenize(toks).decode("utf-8", "replace")
    tokenizer_results.append({"text": text, "tokens": toks, "decoded": detok})
    print(f"ENC {text!r} -> {toks}")

# special ids
print("BOS:", llm.token_bos(), "EOS:", llm.token_eos())

# --- reference logits for a prompt -----------------------------------------
# feed the prompt and capture logits at the final position (and per-position
# via eval of prefixes).
prompt = "1+1="
prompt_toks = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
print("prompt tokens:", prompt_toks)

per_step_logits = []
# llama-cpp-python eval(tokens) returns the logits after processing the
# whole token list; prefix-by-prefix captures per-position logits.
for i in range(1, len(prompt_toks) + 1):
    llm.reset()
    logits = llm.eval(prompt_toks[:i])
    per_step_logits.append(np.asarray(logits, dtype=np.float32).reshape(-1))

# top-5 tokens per step + full logits saved to files
step_tops = []
for i, logits in enumerate(per_step_logits):
    top5 = np.argsort(logits)[::-1][:5]
    step_tops.append(
        {
            "step": i,
            "top5_ids": [int(t) for t in top5],
            "top5_values": [float(logits[t]) for t in top5],
        }
    )
    print(f"step {i}: top5 {top5} {[float(logits[t]) for t in top5]}")

# also dump greedy next-token for a longer context
greedy_next = int(np.argmax(per_step_logits[-1]))
print("greedy next token:", greedy_next, llm.detokenize([greedy_next]).decode())

with open("reference_tokenizer.json", "w") as f:
    json.dump(
        {
            "tests": tokenizer_results,
            "bos": int(llm.token_bos()),
            "eos": int(llm.token_eos()),
        },
        f,
        ensure_ascii=False,
        indent=1,
    )
np.save("reference_logits_prompt.npy", np.stack(per_step_logits))
print("saved reference_tokenizer.json and reference_logits_prompt.npy")
print("DONE")
