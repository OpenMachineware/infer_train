# InferTrain

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CLA assistant](https://cla-assistant.io/readme/badge/OpenMachineware/infer_train)](https://cla-assistant.io/allowlist/OpenMachineware/infer_train)

[![Chinese Docs](https://img.shields.io/badge/Chinese_Docs-Click_here-blue?style=for-the-badge)](./README_zh.md)

**Inference + Training Engine** — a from-scratch LLM inference *and* training engine written in pure Mojo, on top of a minimal C runtime-helper layer (pthread task pool + `mmap`).

```
┌─────────────────────────────── infer_train ───────────────────────────────┐
│                                                                           │
│  it-cli                           it-server (OpenAI-compatible HTTP)      │
│  -m M -p P -n N -sm layer --rpc   /v1/models /v1/chat/completions         │
│  it-rpc-server (RPC worker)       /v1/completions (SSE) /v1/finetune      │
│  --infer-train-mode off|lora|full it-server quantize                      │
│                          │                              │                │
│  ┌───────────────────────▼──────────────────────────────▼──────────────┐  │
│  │  Python bindings (C ABI)  ·  torch.compile backend "infer_train"     │  │
│  └───────────────────────┬──────────────────────────────┬──────────────┘  │
│                          │                              │                 │
│  ┌───────────────────────▼───────────────┐  ┌───────────▼──────────────┐  │
│  │  runtime: Model / generate /          │  │  training: TrainModel /   │  │
│  │  finetune (on-the-fly)                │  │  run_with_grad / AdamW    │  │
│  └───────────────────────┬───────────────┘  └───────────┬──────────────┘  │
│                          │                              │                 │
│  ┌───────────────────────▼──────────────────────────────▼──────────────┐  │
│  │  core: tensor / graph / interpreter / optimizer(IR) / JIT           │  │
│  │  ops: cpu kernels · fused kernels · quantized (GGUF Q4_K/Q5_K/Q6_K/ │  │
│  │        Q8_0/IQ4_NL/IQ4_XS/NF4) · attention (KV cache: dense/paged/  │  │
│  │        sliding window) · autograd                                   │  │
│  │  transformer: qwen2 · hunyuan-dense · qwen35 (Gated DeltaNet hybrid)│  │
│  │  tokenizers: Qwen / Llama / Hunyuan (GPT-2 byte-level BPE, auto-    │  │
│  │             selected from GGUF metadata) + custom registry          │  │
│  │  storage: GGUF loader (mmap) · .mmdl checkpoints (GGUF-compatible)  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# build everything and run the full test suite
make test-m7

# generate with the 1.5B model
make cli
./it-cli -m DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf \
    -p "<｜User｜>What is 1+1?<｜Assistant｜><think>" -n 120 --seed 7

# 7B translation model (Hy-MT2, hunyuan-dense)
./it-cli -m Hy-MT2-7B-Q4_K_M.gguf \
    -p "Translate to English: 今天天气很好。" -n 64

# 27B hybrid (Qwen3.8, Gated DeltaNet + full attention)
./it-cli -m Qwen3.8-27B-UD-Q5_K_M.gguf -c 32768 -p "Hello world" -n 64

# OpenAI-compatible HTTP server
make server
./it-server -m DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf --port 8080
curl http://127.0.0.1:8080/v1/completions \
    -d '{"prompt":"1+1=","max_tokens":32,"temperature":0.6}'

# OpenAI-compatible HTTP server (Python wrapper, concurrent)
INFERTRAIN_MODEL=DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf \
    python -m infer_train.server --port 8080

# Python API
from infer_train import load_model
m = load_model("DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf")
print(m.generate("1+1=", max_tokens=32, seed=7))
losses = m.finetune("What is 1+1?", "1+1 equals 2.", lr=1e-5)  # on-the-fly fine-tuning

# post-fine-tuning re-quantization (.mmdl -> Q4_K_M / Q8_0 / NF4)
./it-server quantize -i model.mmdl -o model_quantized.gguf -f Q4_K_M
```

## GGUF Split Files (Multi-Part Models)

Large models are often distributed as *split* GGUF files — several parts
named `<base>.gguf-NNNNN-of-NNNNN.gguf` (5-digit zero-padded), produced by
llama.cpp's `llama-gguf-split`. InferTrain loads them transparently:

```bash
# split a model (brew install llama.cpp)
llama-gguf-split --split-max-size 600M model.gguf model.gguf
# -> model.gguf-00001-of-00003.gguf  model.gguf-00002-of-00003.gguf  model.gguf-00003-of-00003.gguf

# load any part — all parts are memory-mapped and merged automatically
./it-cli -m model.gguf-00001-of-00003.gguf -p "Hello" -n 64

# Python API
m = load_model("model.gguf-00001-of-00003.gguf")
```

How it works:

* Each part is a valid GGUF file holding only its own tensor subset; the
  **first** part carries the full metadata (the others only `split.no` /
  `split.tensors.count` / `split.count`), and every tensor's offset is
  relative to its own part's data section.
* `load_gguf` auto-detects the input: a part path
  (`…gguf-NNNNN-of-NNNNN.gguf`) loads the whole split; a plain `.gguf` loads
  as a single file; a base name whose part files exist on disk is expanded
  to the split.
* Every tensor is tagged with the part that owns its bytes
  (`GGUFTensor.file_idx`), so dequantization reads from the correct mapping —
  a split model is numerically identical to the single file.

`make test-gguf-split` verifies this: it loads the 1.5B model both as a
single file and as a 3-part split and checks that dequantized weights are
byte-identical (`max_diff = 0.0`) across all parts.

## Multi-Process / Multi-Machine RPC

The engine can be distributed across processes (one machine) or machines
with command-line usage: a separate worker binary(`it-rpc-server`) plus the
`-sm` / `--rpc` flags on `it-cli`.

```bash
# one worker per process / machine (each mmaps the same GGUF file):
./it-rpc-server -m model.gguf --port 50052 &
./it-rpc-server -m model.gguf --port 50053 &

# split the layers across the workers and generate:
./it-cli -m model.gguf -sm layer \
    --rpc 127.0.0.1:50052 --rpc 127.0.0.1:50053 \
    -p "Hello" -n 64
```

How it works:

* `-sm layer` (layer split): the master (`it-cli`) keeps the embedding +
  output head and the generation loop; each `--rpc` endpoint owns a
  contiguous layer range (28 layers over 3 workers → 10/9/9) plus the KV /
  SSM state of those layers.  Per token the master chains the workers:
  `embedding → worker 0 → worker 1 → … → output head → sample`.
* The wire protocol is plain TCP with 4-byte length-prefixed messages
  (`INIT` / `FORWARD` / `RESET` / `PING`); the fp16 hidden state crosses
  the wire as raw bit patterns, so a distributed run is **numerically
  identical** to the local one.
* `--rpc` is repeatable and accepts `host:port`, so the same command line
  works across machines (point the endpoints at the workers' addresses).
* `-sm row` (row-parallel) is accepted but not implemented yet.

`make test-rpc` verifies this locally: it starts two workers on localhost,
runs the same greedy generation once locally and once across the workers,
and requires the outputs to match exactly.

## Core Features

| Feature | Description |
|---|---|
| **Inference** | Three architectures (qwen2 / hunyuan-dense / qwen35 hybrid SSM), direct GGUF weight loading (Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL/IQ4_XS), split/multi-part GGUF (`llama-gguf-split`, auto-detected), KV Cache (dense / Paged / sliding window / adaptive memory), llama.cpp-compatible sampling |
| **Training** | Backward operators, `run_with_grad` automatic differentiation, AdamW/SGD, `train_step`/`eval_step`, gradient accumulation, mixed precision (AMP), dynamic quantization |
| **Fine-tuning** | `finetune_step` (Mojo) and `Model.finetune` (Python) — no service downtime, LoRA-style updates on trainable subsets |
| **Quantization** | Post-fine-tuning re-quantization: FP16 → Q4_K_M / Q8_0 / NF4 (`it-server quantize`), compatible with GGUF toolchain |
| **Storage** | Private `.mmdl` checkpoints: weights + gradients + AdamW m/v + training metadata; incremental updates, delta appends, checkpoint resumption; `strip_to_gguf` exports pure weights |
| **Tokenization** | One generic engine per algorithm (GPT-2 byte-level BPE; SentencePiece to follow) — model families differ only in data (flavor tag, added tokens, bos/eos), auto-selected by `tokenizer.ggml.model`/`tokenizer.ggml.pre`, `register_tokenizer` for custom tokenizers |
| **Distribution** | Multi-process / multi-machine RPC: `it-rpc-server` workers + `-sm layer` / `--rpc host:port` on `it-cli`, layer-split with per-worker KV/SSM state, numerically identical to the local run |
| **API** | C ABI + Python bindings + `torch.compile` backend; OpenAI-compatible HTTP endpoints (`/v1/models`, `/v1/chat/completions`, `/v1/completions` SSE, `/v1/finetune`, `/v1/finetune/status`), `INFERTRAIN_API_KEY` authentication |
| **CLI** | Three entry points over one shared core: `it-cli` (quick verification: `-m -p -c -n --temp --top-p --top-k --repeat-penalty -t --seed -sm --rpc`), `it-server` (OpenAI HTTP service + `quantize`), `it-rpc-server` (RPC worker); all take the `--infer-train-*` parameter group |

## Performance Benchmarks (Apple M1 Max, 64 GB; CPU-only, no GPU)

Prompt: `The quick brown fox jumps over the lazy dog.` (token count varies by tokenizer: 26 / 27 / 26 / 18 for the four models), 16-token generation (8 for the 35B), temp 0.6 / top-k 40 / top-p 0.95, seed 7. InferTrain uses the **default Q4-resident path** (weights stay packed in RAM; its matmul hot path is **multithreaded** - M12 splits the N output columns across a C thread pool, each thread dequantizing into its own private per-block scratch). llama.cpp is the **CPU-only** build (`llama.cpp-0.4.0/build-cpu`, `GGML_METAL=OFF`) at 8 threads, measured with `llama-bench` (warmup + repeats). The 35B uses a 128-token context; the rest use 512.

| Model | Size | InferTrain Prefill | InferTrain Decode | InferTrain Memory | llama.cpp Prefill | llama.cpp Decode | Decode Ratio |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B（Q4_K_XL） | 0.38 GB | 24.78 t/s | 8.99 t/s | 0.60 GiB | 858.73 t/s | 195.39 t/s | 4.6% |
| DeepSeek-R1-Distill-Qwen-1.5B（Q5_K_M） | 1.20 GB | 13.99 t/s | 5.48 t/s | 1.29 GiB | 316.08 t/s | 86.78 t/s | 6.3% |
| Hy-MT2-7B（Q4_K_M, hunyuan-dense） | 4.31 GB | 4.92 t/s | 1.87 t/s | 4.41 GiB | 90.00 t/s | 23.76 t/s | 7.8% |
| Qwen3.8-27B（Q5_K_M, qwen35） | 18.0 GB | 0.96 t/s | 0.37 t/s | 18.97 GiB | 17.87 t/s | 4.86 t/s | 7.6% |
| Qwen3.6-35B-A3B（Q5_K_M, qwen35moe） | 23.6 GB | 2.64 t/s | 1.63 t/s | 7.25 GiB | 58.03 t/s | 22.24 t/s | 7.3% |

> **Q4-resident vs fp16**: the default Q4-resident path (M11) keeps weights packed (low memory); its matmul is now multithreaded (M12). The fp16 path is still faster (e.g. 1.5B decode 8.04 t/s vs 3.28) at ~3.4× the memory (1.5B: 4.36 GiB vs 1.29 GiB).
>
> ⚠️ **Performance target not met**: decode is 4–30% of llama.cpp (CPU, 8 threads) — 4.6–7.8%. Numerical correctness is unaffected (token-identical with llama.cpp); the remaining gap after the M12 multithreaded Q4 matmul is kernel efficiency: per-block dequantization without AMX-style weight repacking, the scalar DeltaNet recurrence, element-wise attention, and serial MoE routing. Full analysis, 32K context memory data, and the optimization roadmap are in `docs/M7_PERFORMANCE_REPORT.md`; M5/M6 data in `docs/M5_PERFORMANCE_REPORT.md` and `docs/M6_TRAINING_REPORT.md`.
>

Reproduce: `make bench_cpu && ./bench_cpu MODEL.gguf` (InferTrain; optional args `[q4|fp16] [n_predict] [n_warmup] [ctx]`; the 35B uses `q4 8 8 128`), and `./llama.cpp-0.4.0/build-cpu/bin/llama-bench -m MODEL.gguf -t 8 -p <prompt_tokens> -n <gen_tokens>` (llama.cpp; 35B uses `-p 18 -n 8`).

## Verified Numerical Correctness

* **Hy-MT2-7B (hunyuan-dense)**: Token-identical with llama.cpp — 4-step greedy decoding yields identical tokens, full logits vector correlation ≥ 0.9995.
* **Qwen3.8-27B (qwen35 hybrid)**: 1-token / 2-token / 6-step greedy decoding matches llama.cpp token-for-token (single-token logits correlation 0.9991).
* **1.5B (qwen2)**: M3 regression baseline `reference_logits_5.npy` bitwise identical (post-refactor).
* All GGUF dequantization formats verified against gguf-py (llama.cpp official Python implementation). The FP32 dequantization path (`dequantize_into_f32` / `dequantize_q4_k_m_f32`) is **bit-exact** with llama.cpp's `ggml-quants.c` (0/1024 mismatches across Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL/IQ4_XS/F32); the fp16 inference path stores one fp16 rounding of the exact value (max rel. error 4.85e-4, bounded by the fp16 half-ULP 2⁻¹¹ ≈ 4.88e-4) — analysis and training impact in `docs/M7_PERFORMANCE_REPORT.md` §4.
* M6 training: bitwise identical with PyTorch eager (loss 3.58 → 1.77).

## Ecosystem

* [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines (Chinese: [CONTRIBUTING_zh.md](CONTRIBUTING_zh.md))
* [CLA.md](CLA.md) — Contributor License Agreement (Chinese: [CLA_zh.md](CLA_zh.md))
* [docs/NEW_OPERATOR_GUIDE.md](docs/NEW_OPERATOR_GUIDE.md) — How to discover and add new operators
* [docs/FUSION_GUIDE.md](docs/FUSION_GUIDE.md) — How to extend operator fusion passes
* [docs/PORTING_GUIDE.md](docs/PORTING_GUIDE.md) — How to port to new hardware platforms (CUDA/ROCm/…)
* [docs/M5_PERFORMANCE_REPORT.md](docs/M5_PERFORMANCE_REPORT.md) / [docs/M6_TRAINING_REPORT.md](docs/M6_TRAINING_REPORT.md) — Milestone reports
* [LICENSE](LICENSE)

## Development

```bash
make tp          # build C runtime helpers (thread pool + mmap)
make test        # M1/M2 core tests
make test-m3     # tokenizer / ops / forward validation
make test-gguf-split  # GGUF split-file (multi-part) loading
make test-m5     # IR optimizer + JIT + torch.compile backend
make test-m6     # training suites
make test-m7     # everything + the M7 feature suites
make test-rpc    # multi-process RPC (two localhost workers, -sm layer)
make cli         # the it-cli binary
make server      # the it-server binary
make rpc-server  # the it-rpc-server worker binary
make bench_cpu   # the CPU benchmark binary (README table; bench_cpu <model.gguf>)
make bench_pool  # the C thread pool per-submission overhead micro-benchmark
```

Test composition: Mojo executables (`tests/*.mojo`, compiled with `pixi run mojo build -I .`) + Python acceptance suites (`tests/python/`). Mojo 1.0 constraints: `def`-only, no runtime globals, `fn` deprecated; C-side helpers linked via `-Xlinker` (`tools/thread_pool.c`).

## Known Limitations / TODO

* **Qwen3.8 MTP (nextn_predict) module not enabled** — matches llama.cpp default single-model decoding; MTP speculative decoding deferred to future version.
* **NF4 is a private GGML extended type (30)** — llama.cpp cannot read NF4 weights; export to llama.cpp using `-f Q4_K_M` / `Q8_0`.
* **fp16 dequantization precision gap** — inference weights are the fp16 rounding of the exact (FP32) dequantized value: ≤ 4.88e-4 relative per element, deterministic, negligible for inference (token-identical) and for fine-tuning (far below the re-quantization error on export), but it breaks bit-reproducibility against FP32-initialized runs. The bit-exact FP32 path (`dequantize_into_f32` / `dequantize_q4_k_m_f32`) is in place; future version directions: quantized dot kernels (closes the gap entirely), FP32 fine-tune initialization, per-tensor fp32 option — see `docs/M7_PERFORMANCE_REPORT.md` §4.
* Mixed precision training (AMP) uses fp32 master weights + fp16 shadow; row-parallel (`-sm row`, tensor parallelism with allreduce) is not implemented yet — the RPC transport and layer split (`-sm layer`) are in place for it.
* GPU backends (CUDA/ROCm) interfaces are ready (see PORTING_GUIDE), kernels remain for porting.
* More ops / more models (MoE, VL), more platforms (Windows/Linux packaging).
