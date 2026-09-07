# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_cpu.mojo
#
# CPU benchmark for one GGUF model: prefill + decode throughput and peak
# resident memory, for the README's llama.cpp comparison table.
#
# Methodology (matches docs/M7_PERFORMANCE_REPORT.md §1):
#   * prompt: "The quick brown fox jumps over the lazy dog." (10 tokens)
#   * 16-token generation, temp 0.6 / top-k 40 / top-p 0.95, seed 7
#   * 512-token context
#   * one warmup pass (mmap page faults + JIT) then one measured pass;
#     the KV/SSM cache is reset between the two
#   * decode always runs exactly 16 steps (no early EOS stop), so the
#     number matches llama-bench's tg16
#
# Usage: bench_cpu <model.gguf>
#
# Build:
#   make tp
#   pixi run mojo build -I . tools/bench_cpu.mojo \
#       -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib -o bench_cpu

from src.core.gguf_loader import load_gguf
from src.core.memory import process_resident_bytes
from src.core.ops.attention.kv_cache import KVCacheType
from src.core.ops.base.op_registry import OpRegistry
from src.core.sampler import Sampler, sample_dynamic, seed_sampler
from src.core.thread_pool import now_ns
from src.core.tensor import Tensor
from src.core.tokenizers import make_tokenizer
from src.core.transformer import (
    TransformerModel,
    build_graph,
    collect_weights,
    load_config,
)
from src.runtime.inference import Model
from std.sys import argv
from std.utils.static_tuple import StaticTuple

comptime PROMPT: String = "The quick brown fox jumps over the lazy dog."
comptime N_PREDICT: Int = 16
comptime CTX: Int = 512


def load_model_qr(
    path: String, ctx_len: Int, quant_resident: Bool
) raises -> Model:
    """load_model with an explicit quant_resident flag (default True)."""
    var ctx = load_gguf(path)
    var config = load_config(ctx)
    var weights = collect_weights(ctx)
    var model = TransformerModel(
        config, ctx^, ctx_len, quant_resident=quant_resident
    )
    model.weights = weights^
    var tokenizer = make_tokenizer(model.ctx, String(""))
    var graph = build_graph(model)
    var registry = OpRegistry()
    return Model(model^, tokenizer^, registry^, graph^)


def _forward_prompt(
    mut model: Model, tokens: List[Int]
) raises -> Tensor[DType.float32, 1]:
    """Feed the prompt one token at a time (the engine's prefill path)."""
    var logits = model.transformer.forward(tokens[0], 0)
    var i = 1
    while i < len(tokens):
        logits = model.transformer.forward(tokens[i], i)
        i += 1
    return logits


def _decode(
    mut model: Model,
    mut tokens: List[Int],
    mut logits: Tensor[DType.float32, 1],
    sampler: Sampler,
    n: Int,
) raises -> Int:
    """Exactly n decode steps (no EOS stop); returns the steps run."""
    var steps = 0
    while steps < n:
        var t = sample_dynamic[DType.float32](logits, sampler, tokens)
        tokens.append(t)
        logits = model.transformer.forward(t, len(tokens) - 1)
        steps += 1
    return steps


def _gib(x: Int) -> Float64:
    return Float64(x) / (1024.0 * 1024.0 * 1024.0)


def _atoi(s: String, default: Int) -> Int:
    var n = 0
    var ok = True
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c >= 48 and c <= 57:
            n = n * 10 + (c - 48)
        else:
            ok = False
    return n if ok and n > 0 else default


def main() raises:
    var arg_list = List[String]()
    for a in argv():
        arg_list.append(String(a))
    if len(arg_list) < 2:
        print("usage: bench_cpu <model.gguf> [q4|fp16] [n_predict] [n_warmup] [ctx]")
        raise Error("missing model path")
    var path = arg_list[1]
    var quant_resident = True
    if len(arg_list) >= 3:
        quant_resident = (arg_list[2] != "fp16")
    var n_predict = N_PREDICT
    if len(arg_list) >= 4:
        n_predict = _atoi(arg_list[3], N_PREDICT)
    var n_warmup = N_PREDICT
    if len(arg_list) >= 5:
        n_warmup = _atoi(arg_list[4], N_PREDICT)
    var ctx = CTX
    if len(arg_list) >= 6:
        ctx = _atoi(arg_list[5], CTX)

    var mem0 = process_resident_bytes()
    var model = load_model_qr(path, ctx, quant_resident)
    var cfg = model.transformer.config
    print("weight mode:  ", "q4-resident" if quant_resident else "fp16 (threaded)")

    var tokens = model.tokenizer.encode_with_bos(PROMPT)
    seed_sampler(7)
    var sampler = Sampler(
        temperature=Float32(0.6), top_k=40, top_p=Float32(0.95)
    )

    # warmup pass: page faults + first-run costs (discarded)
    var warm = List[Int]()
    warm.append(tokens[0])
    var i = 1
    while i < len(tokens):
        warm.append(tokens[i])
        i += 1
    var wlogits = _forward_prompt(model, warm)
    _ = _decode(model, warm, wlogits, sampler, n_warmup)
    model.transformer.reset_cache()

    # measured pass
    var t0 = now_ns()
    var logits = _forward_prompt(model, tokens)
    var t1 = now_ns()
    var steps = _decode(model, tokens, logits, sampler, n_predict)
    var t2 = now_ns()

    var prefill_ns = t1 - t0
    var decode_ns = t2 - t1
    var prefill_tps = Float64(len(tokens)) * 1e9 / Float64(prefill_ns)
    var decode_tps = Float64(steps) * 1e9 / Float64(decode_ns)
    var mem1 = process_resident_bytes()

    print("model:        ", path)
    print("arch:         ", cfg.arch_str, "layers:", cfg.n_layers)
    print("prompt tokens:", len(tokens))
    print("prefill:      ", prefill_ns // 1_000_000, "ms ->", prefill_tps, "t/s")
    print("decode:       ", decode_ns // 1_000_000, "ms ->", decode_tps, "t/s (", steps, "steps )")
    print("resident:     ", _gib(mem1), "GiB (delta", _gib(mem1 - mem0), "GiB)")
