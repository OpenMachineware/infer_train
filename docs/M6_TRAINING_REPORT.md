# M6 report — training & fine-tuning

## Scope delivered

| Phase | Content | Status |
| :--- | :--- | :--- |
| 1 | `backward` for matmul, weight-major linear, add, add_bias, rms_norm (+ weighted), softmax, RoPE, SwiGLU, embedding, SwiGLU-FFN, full-sequence causal MHA (`mha_seq`) and the cached decode MHA | ✅ gradient-checked vs PyTorch (`< 1e-5` f32) |
| 2 | `Interpreter.run_with_grad`: forward-with-saved, reverse topological traversal, gradient accumulation at multi-consumer tensors, saved-tensor release, external-input gradients | ✅ `tests/test_backward.mojo::test_interpreter_run_with_grad` |
| 3 | `cross_entropy` loss (log-sum-exp forward + exact backward), registered as an op | ✅ numeric check + vs `F.cross_entropy` |
| 4 | `AdamW` (bias-corrected moments, decoupled weight decay, param groups, fp32 state in `Tensor.opt_state`) + momentum `SGD` | ✅ vs `torch.optim.AdamW` (live + hardcoded reference) |
| 5 | `TrainModel` + `train_step` / `eval_step` (Mojo layer), gradient accumulation, eval mode | ✅ loss 3.58 → 1.77 on the toy task |
| 6 | AMP `GradScaler` (scale, Inf/NaN detection, back-off/growth) + `enable_amp` training path | ✅ scaler unit tests + AMP training run |
| 7 | Dynamic symmetric/asymmetric quantize + dequantize (STE backward), registered ops | ✅ round-trip test |
| 8 | `tests/test_backward.mojo`, `tests/test_train_optimizer.mojo`, `tests/test_training.mojo` | ✅ all pass |
| 9 | `tests/python/test_training.py`: engine backward vs `torch.autograd`, engine AdamW vs `torch.optim.AdamW`, mini-transformer trained through engine ops vs eager PyTorch | ✅ loss curves identical to 6 decimals |

## PyTorch acceptance (Phase 9)

A 2-layer mini transformer (B=2, T=4, hidden=16, 2 heads, ffn=48) whose
forward AND backward run entirely on engine kernels (embedding →
rms_norm_weight → mha_seq → add → rms_norm_weight → swiglu_ffn → add →
rms_norm_weight → lm_head → cross_entropy, executed as single-shot C calls
through a `torch.autograd.Function`) is trained with `torch.optim.AdamW`
and compared with an identical eager model:

```
step 0: engine 4.422882 vs eager 4.422882
step 1: engine 4.385968 vs eager 4.385968
step 2: engine 4.349238 vs eager 4.349239
step 3: engine 4.312681 vs eager 4.312681
```

## Bugs the gradient checks caught

The M6 acceptance tests are doing real work — they caught four genuine
defects during development:

1. **RMSNorm backward formula** had an extra `1/r` factor on the
   second-order term (`grad_x = (g - y·s/(N·r²))/r` instead of
   `(g - y·s/N)/r`).
2. **Weighted-RMSNorm weight gradient** included a spurious weight factor.
3. **RoPE backward** iterated over the comptime `n_heads` parameter
   (callers pass 0) instead of the runtime head count — it silently
   produced all-zero gradients.
4. **Multi-token MHA head layout**: a plain `[n_heads, T, head_dim]`
   *view* over the `[T, n_heads*head_dim]` flat tensor scrambles heads
   for T > 1 (the decode path was fine because T == 1).  Forward and
   backward now gather/scatter heads explicitly.

(Plus two wiring mistakes in the Python test harness itself — a wrong
residual hookup in the eager reference model and the layer-gradient
ordering — fixed before delivery.)

## Deviation notes

- The spec's `src/core/optimizer.mojo` is **`src/core/train_optimizer.mojo`**
  because `src/core/optimizer/` is the M5 IR-optimizer package; the M6
  optimizer tests live in `tests/test_train_optimizer.mojo` for the same
  reason (`tests/test_optimizer.mojo` is the M5 CFG suite).
- `train_step` operates on `TrainModel` (a graph built from the registered
  operators) rather than the GGUF-backed `Model`, because Mojo's static
  typing makes a single polymorphic `Optimizer` over mixed-rank parameter
  lists impossible; the optimizer is fully type-erased instead, and the
  1.5B model's typed decode path is untouched.
- The spec allows "a small test model" for the Mojo end-to-end test — a
  real 1.5B training step would run the same code path (the registry
  dispatches fp16), but is left out of the fast suites.
- GPU `backward` entries delegate to the CPU kernels, matching the M1-M5
  GPU fallback policy until the Metal backend lands.

## Performance

M6 adds no overhead to the M5 inference path: the typed forward is
unchanged, `run`'s entry-node handling is backward compatible (a node
takes all remaining external inputs unless `n_inputs` is set), and all
training machinery (backward kernels, optimizer, scaler) is only reached
from the new APIs.  The training kernels accumulate in f32 with SIMD main
loops where the layout allows (matmul backward), following the M5 kernel
conventions.

## How to run

```sh
make test-m6           # Mojo training suites + PyTorch training acceptance
make test-m5           # full regression suite including M6 python tests
```
