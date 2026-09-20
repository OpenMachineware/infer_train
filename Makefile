# infer_train build helpers.
#
# Mojo is provided by the pixi environment (see pixi.toml).  Every target
# shells out to `pixi run mojo` so the correct toolchain and MODULAR_HOME are
# picked up automatically.  The C runtime helpers (tools/thread_pool.c:
# pthread task pool + mmap + wall clock) are compiled to a small dylib and
# linked into every build that reaches the engine kernels via `-Xlinker`.

MOJO := pixi run mojo
SRC := src
TP := python/infer_train/_lib/libinfer_train_tp.dylib
TP_XLINK := -Xlinker $(TP)
MWQ := python/infer_train/_lib/libinfer_train_mwq.dylib
# BLAS framework for Apple Silicon
BLAS_XLINK := -Xlinker "-framework" -Xlinker "Accelerate"

.PHONY: test test-m3 test-m4 test-m5 test-m6 test-m7 test-m8 test-gpu \
        test-gguf-split test-gguf test-rpc test-thread-pool clean tp mwq \
        server cli rpc-server infer_train version check-mem bench_cpu bench_pool \
        bench_transformer_core bench_blas bench_dequant bench_simd \
        bench_q8k_vs_fp32 bench_qmatmul_threading test-kernel dump-gguf ref-forward \
        test-q2k-perf test-q3k-perf test-q4k-perf test-q4k-multisize test-q5k-perf test-q5k-multisize test-q6k-perf test-q6k-multisize \
        test-iq4xs-perf

# M12: the Q4-resident matmul pool workers as a standalone Mojo shared
# library.  Mojo 1.0 only honors @export in a build's entry module and
# executables strip unreferenced symbols, so the workers the C pool
# resolves via dlsym must live in a globally-loaded dylib: the pool
# loader (core/thread_pool.mojo) dlopens it RTLD_GLOBAL next to the C
# pool library, in every process.
mwq:
	$(MOJO) build -I . src/core/ops/cpu/mwq_workers.mojo \
		--emit shared-lib -o $(MWQ)

# The C runtime helper library (thread pool + mmap + clock), plus the
# M12 worker dylib (tp depends on mwq so `make tp` builds both).
tp: mwq
	cc -O2 -shared -o $(TP) tools/thread_pool.c

# Generate src/version.mojo from the `version` field of pixi.toml.  Every
# build that compiles an entry (server, cli, rpc-server, infer_train)
# depends on this.
version:
	pixi run python tools/gen_version.py

# Compile and run the M1/M2 example tests (now under tests/, built with
# `-I .` so `src.`-prefixed imports resolve).
test: tp
	$(MOJO) build -I . tests/test_core.mojo $(TP_XLINK) -o tests/test_core && ./tests/test_core
	$(MOJO) build -I . tests/test_quant.mojo $(TP_XLINK) -o tests/test_quant && ./tests/test_quant
	$(MOJO) build -I . tests/test_cpuops.mojo $(TP_XLINK) -o tests/test_cpuops && ./tests/test_cpuops
	$(MOJO) build -I . tests/test_registry.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_registry && ./tests/test_registry
	$(MOJO) build -I . tests/test_e2e.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_e2e && ./tests/test_e2e

# M3 tests (tests/ builds with -I . so `src.` imports resolve).
test-m3: tp
	$(MOJO) build -I . tests/test_json.mojo $(TP_XLINK) -o tests/test_json
	./tests/test_json
	$(MOJO) build -I . tests/test_tokenizer.mojo $(TP_XLINK) -o tests/test_tokenizer
	./tests/test_tokenizer
	$(MOJO) build -I . tests/test_ops.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_ops
	./tests/test_ops
	$(MOJO) build -I . tests/test_sampler.mojo $(TP_XLINK) -o tests/test_sampler
	./tests/test_sampler
	$(MOJO) build -I . tests/test_forward.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_forward
	./tests/test_forward
	$(MAKE) test-gguf-split
	$(MAKE) test-gguf
	$(MAKE) test-gpu

# GPU kernels (Metal).  Falls back to the CPU kernels on machines without a
# Metal GPU, so it is safe to run everywhere.
test-gpu: tp
	$(MOJO) build -I . tests/test_gpuops.mojo $(TP_XLINK) -o tests/test_gpuops
	./tests/test_gpuops
	$(MOJO) build -I . tests/test_gpu_pipeline.mojo $(TP_XLINK) -o tests/test_gpu_pipeline
	./tests/test_gpu_pipeline
	$(MOJO) build -I . tests/test_gpu_weight_proj.mojo $(TP_XLINK) -o tests/test_gpu_weight_proj
	./tests/test_gpu_weight_proj
	$(MOJO) build -I . tests/test_tiled_matmul.mojo $(TP_XLINK) -o tests/test_tiled_matmul
	./tests/test_tiled_matmul
	$(MOJO) build -I . tests/bench_dynamic_dispatch.mojo $(TP_XLINK) -o tests/bench_dynamic_dispatch
	./tests/bench_dynamic_dispatch
	$(MOJO) build -I . tests/test_q4k_gpu.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_q4k_gpu
	./tests/test_q4k_gpu
	$(MOJO) build -I . tests/test_k_quant_gpu.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_k_quant_gpu
	./tests/test_k_quant_gpu
	# GPU pipeline tests (Phase 1-5)
	$(MOJO) build -I . tests/test_gpu_forward_pipeline.mojo $(TP_XLINK) -o tests/test_gpu_forward_pipeline
	./tests/test_gpu_forward_pipeline
	$(MOJO) build -I . tests/bench_gpu_decode_pipeline.mojo $(TP_XLINK) -o tests/bench_gpu_decode_pipeline
	./tests/bench_gpu_decode_pipeline
	$(MOJO) build -I . tests/profile_gpu_kernels.mojo $(TP_XLINK) -o tests/profile_gpu_kernels
	./tests/profile_gpu_kernels

# GGUF split-file (multi-part) loading.  Needs the split part files next to
# the repo root (see tests/test_gguf_split.mojo); reports SKIP if absent.
test-gguf-split: tp
	$(MOJO) build -I . tests/test_gguf_split.mojo $(TP_XLINK) -o tests/test_gguf_split
	./tests/test_gguf_split

# M11: Q4-resident weights - the 27B memory budget (< 25 GB after load +
# full page touch) and the quantized-vs-dequantized forward match.  Needs
# the model files next to the repo root; reports SKIP if absent.
test-gguf: tp
	$(MOJO) build -I . tests/test_gguf.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_gguf
	./tests/test_gguf

# M5: the optimizer/CFG/JIT suites (Mojo executables).
test-m5-mojo: tp
	$(MOJO) build -I . tests/test_optimizer.mojo $(TP_XLINK) -o tests/test_optimizer
	./tests/test_optimizer
	$(MOJO) build -I . tests/test_jit.mojo $(TP_XLINK) -o tests/test_jit
	./tests/test_jit

# M8: SIMD adaptive width + JIT shape specialization (CPU path).
test-m8: tp
	$(MOJO) build -I . tests/test_simd_utils.mojo $(TP_XLINK) -o tests/test_simd_utils
	./tests/test_simd_utils
	$(MOJO) build -I . tests/test_jit_cache.mojo $(TP_XLINK) -o tests/test_jit_cache
	./tests/test_jit_cache

# M6: training - backward gradient checks, the AdamW/SGD optimizers, the
# Mojo training loop, and the PyTorch training acceptance suite.
test-m6-mojo: tp
	$(MOJO) build -I . tests/test_backward.mojo $(TP_XLINK) -o tests/test_backward
	./tests/test_backward
	$(MOJO) build -I . tests/test_train_optimizer.mojo $(TP_XLINK) -o tests/test_train_optimizer
	./tests/test_train_optimizer
	$(MOJO) build -I . tests/test_training.mojo $(TP_XLINK) -o tests/test_training
	./tests/test_training

test-m6-python: tp
	$(MOJO) build -I . src/bindings/infer_train_bindings.mojo $(TP_XLINK) \
		--emit shared-lib -o python/infer_train/_lib/libinfer_train.dylib
	pixi run python -m pytest tests/python/test_training.py -v

# M4+M5: build the C-API shared library and run the Python/PyTorch suite.
# The e2e 1.5B generation test needs the GGUF next to the repo root.
test-m4: tp
	$(MOJO) build -I . src/bindings/infer_train_bindings.mojo $(TP_XLINK) \
		--emit shared-lib -o python/infer_train/_lib/libinfer_train.dylib
	pixi run python -m pytest tests/python/ -v

# M7: tokenizer abstraction, dequantizers, mmdl, finetune, KV cache,
# requantize + the Python finetune/server API tests.
# (test_dequant_m7 compares against gguf-py-generated reference bins; the
# generator falls back to a bundled numpy reference when gguf-py is absent.)
test-m7-mojo: tp
	pixi run python tools/gen_dequant_refs.py
	$(MOJO) build -I . tests/test_tokenizer_m7.mojo $(TP_XLINK) -o tests/test_tokenizer_m7
	./tests/test_tokenizer_m7
	$(MOJO) build -I . tests/test_dequant_m7.mojo $(TP_XLINK) -o tests/test_dequant_m7
	./tests/test_dequant_m7
	$(MOJO) build -I . tests/test_batch_prefill.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_batch_prefill
	./tests/test_batch_prefill
	$(MOJO) build -I . tests/test_mmdl.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_mmdl
	./tests/test_mmdl
	$(MOJO) build -I . tests/test_finetune.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_finetune
	./tests/test_finetune
	$(MOJO) build -I . tests/test_kv_cache_m7.mojo $(TP_XLINK) -o tests/test_kv_cache_m7
	./tests/test_kv_cache_m7
	$(MOJO) build -I . tests/test_kv_cache_quantization.mojo $(TP_XLINK) -o tests/test_kv_cache_quantization
	./tests/test_kv_cache_quantization
	$(MOJO) build -I . tests/test_requantize.mojo $(TP_XLINK) -o tests/test_requantize
	./tests/test_requantize

test-m7-python: tp
	$(MOJO) build -I . src/bindings/infer_train_bindings.mojo $(TP_XLINK) \
		--emit shared-lib -o python/infer_train/_lib/libinfer_train.dylib
	pixi run python -m pytest tests/python/test_finetune.py tests/python/test_server.py -v

# Multi-model validation (needs the 7B / 27B / 35B / Qwen3-0.6B GGUFs next
# to the repo root; each test SKIPs when its model file is absent).
test-m7-models: tp
	$(MOJO) build -I . tests/test_hunyuan.mojo $(TP_XLINK) -o tests/test_hunyuan
	./tests/test_hunyuan
	$(MOJO) build -I . tests/test_qwen3.mojo $(TP_XLINK) -o tests/test_qwen3
	./tests/test_qwen3
	$(MOJO) build -I . tests/test_qwen35moe.mojo $(TP_XLINK) -o tests/test_qwen35moe
	./tests/test_qwen35moe

test-m5: test test-m3 test-m5-mojo test-m4

# M6: everything (Mojo training suites + the PyTorch acceptance tests).
test-m6: test-m6-mojo test-m6-python

# M7: everything (regression suites + the M7 feature suites).
test-m7: test test-m3 test-m5-mojo test-m6-mojo test-m7-mojo test-m7-python

# M10: the it-server binary (HTTP service.
# Shared code: src/core/cli_common + src/core/http.
server: tp version
	$(MOJO) build -I . src/core/server-cli/it_server.mojo $(TP_XLINK) -o it-server

# M10: the it-cli binary (quick-verification CLI).
cli: tp version
	$(MOJO) build -I . src/core/cli/it_cli.mojo $(TP_XLINK) $(BLAS_XLINK) -o it-cli

# M8: the it-rpc-server worker binary, split out of the main CLI in M10.
rpc-server: tp version
	$(MOJO) build -I . src/core/server-cli/it_rpc_server.mojo $(TP_XLINK) \
		-o it-rpc-server

# Legacy alias: the old `infer_train` binary name now builds the it-server
# entry (same code, different output name).
infer_train: tp version
	$(MOJO) build -I . src/core/server-cli/it_server.mojo $(TP_XLINK) -o infer_train

# M11: the Q4-resident memory verification tool (check_mem <model.gguf>).
check-mem: tp
	$(MOJO) build -I . tools/check_mem.mojo $(TP_XLINK) -o tools/check_mem

# M12: the CPU benchmark for the README's llama.cpp comparison table
# (bench_cpu <model.gguf> [q4|fp16] [n_predict] [n_warmup] [ctx]).
bench_cpu: tp mwq
	$(MOJO) build -I . tools/bench_cpu.mojo $(TP_XLINK) -Xlinker $(MWQ) $(BLAS_XLINK) -o bench_cpu

# M12: the C thread pool's per-submission overhead micro-benchmark
# (bench_pool [nthreads]; noop worker, no model needed).
bench_pool: tp
	$(MOJO) build -I . tools/bench_pool.mojo $(TP_XLINK) -o bench_pool

# Pure compute benchmark for stories15M model (FP32, no quantization).
# Measures core transformer throughput (bench_transformer_core <stories15M.bin>).
bench_transformer_core:
	$(MOJO) build -I . tools/bench_transformer_core.mojo -o bench_transformer_core

# BLAS vs SIMD matmul benchmark (Accelerate framework).
bench_blas:
	$(MOJO) build -I . tools/bench_blas.mojo $(BLAS_XLINK) -o bench_blas

# Dequantization and quantized matmul microbenchmark.
bench_dequant: tp
	$(MOJO) build -I . tools/bench_dequant.mojo $(TP_XLINK) -o bench_dequant

# SIMD fused dot product benchmark.
bench_simd: tp
	$(MOJO) build -I . tools/bench_simd.mojo $(TP_XLINK) -o bench_simd

# Q8_K + SDOT vs FP32 SIMD comparison benchmark.
bench_q8k_vs_fp32: tp
	$(MOJO) build -I . tools/bench_q8k_vs_fp32.mojo $(TP_XLINK) $(TP_XLINK) -o bench_q8k_vs_fp32

# Q8_K + SDOT threaded benchmark.
bench_q8k_sdot_threaded: tp
	$(MOJO) build -I . tools/bench_q8k_sdot_threaded.mojo $(TP_XLINK) $(TP_XLINK) -o bench_q8k_sdot_threaded

# Quantized matmul threading benchmark (pthread pool scaling).
bench_qmatmul_threading: tp
	$(MOJO) build -I . tools/bench_qmatmul_threading.mojo $(TP_XLINK) $(TP_XLINK) -o bench_qmatmul_threading

# Forward breakdown benchmark (per-layer timing analysis).
bench_forward_breakdown: tp
	$(MOJO) build -I . tests/bench_forward_breakdown.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/bench_forward_breakdown

# GPU decode overhead breakdown benchmark.
bench_gpu_overhead: tp
	$(MOJO) build -I . tests/bench_gpu_overhead.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/bench_gpu_overhead

# Matmul micro-benchmark (per-weight GFLOPS measurement).
bench_matmul_micro: tp mwq
	$(MOJO) build -I . tests/bench_matmul_micro.mojo $(TP_XLINK) -Xlinker $(MWQ) $(BLAS_XLINK) -o tests/bench_matmul_micro

# Qwen3-0.6B end-to-end benchmark (prefill + decode t/s).
bench_qwen3: tp mwq
	$(MOJO) build -I . tests/bench_qwen3.mojo $(TP_XLINK) -Xlinker $(MWQ) $(BLAS_XLINK) -o tests/bench_qwen3

# Q4_K matmul micro-benchmark (single kernel GFLOPS).
bench_q4k_matmul: tp
	$(MOJO) build -I . tests/bench_q4k_matmul.mojo $(TP_XLINK) -o tests/bench_q4k_matmul

# Hunyuan model end-to-end benchmark.
bench_hunyuan: tp mwq
	$(MOJO) build -I . tests/bench_hunyuan.mojo $(TP_XLINK) -Xlinker $(MWQ) $(BLAS_XLINK) -o tests/bench_hunyuan

# Decode performance breakdown benchmark.
bench_decode_breakdown: tp mwq
	$(MOJO) build -I . tests/bench_decode_breakdown.mojo $(TP_XLINK) -Xlinker $(MWQ) $(BLAS_XLINK) -o tests/bench_decode_breakdown

# GPU decode benchmark with FP16 pre-dequantization.
bench_decode_gpu: tp
	$(MOJO) build -I . tests/bench_decode_gpu.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/bench_decode_gpu

# Test GPU FFN
test_gpu_ffn: tp
	$(MOJO) build -I . tests/test_gpu_ffn.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_gpu_ffn

# Benchmark GPU FFN end-to-end
bench_gpu_ffn_e2e: tp
	$(MOJO) build -I . tests/bench_gpu_ffn_endtoend.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/bench_gpu_ffn_e2e

# Benchmark GPU vs CPU
bench_gpu_vs_cpu: tp
	$(MOJO) build -I . tests/bench_gpu_vs_cpu.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/bench_gpu_vs_cpu

# M8: multi-process RPC test - two localhost workers, -sm layer, output
# must match the single-process run exactly (needs the 1.5B GGUF at the
# repo root; SKIPs when absent).
test-rpc: cli rpc-server
	bash tools/test_rpc.sh

# M9: the Mojo-native work-stealing CPU thread pool (runtime init,
# correctness, exactly-once, and all-threads-loaded checks).
test-thread-pool: tp
	$(MOJO) build -I . tests/test_thread_pool.mojo $(TP_XLINK) \
		-o tests/test_thread_pool
	./tests/test_thread_pool

# K-quant vec_dot kernel tests
test-kernel: tp
	$(MOJO) build -I . tests/test_q4k_kernel.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_q4k_kernel
	./tests/test_q4k_kernel
	$(MOJO) build -I . tests/test_q5k_kernel.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_q5k_kernel
	./tests/test_q5k_kernel

# Q2_K performance comparison test
test-q2k-perf: tp
	$(MOJO) build -I . -O3 tests/test_q2k_mojo_real.mojo $(TP_XLINK) -o tests/test_q2k_mojo_real
	./tests/test_q2k_mojo_real

# Q3_K performance test
test-q3k-perf: tp
	$(MOJO) build -I . -O3 tests/test_q3k_mojo_real.mojo $(TP_XLINK) -o tests/test_q3k_mojo_real
	./tests/test_q3k_mojo_real

# Q4_K performance test
test-q4k-perf: tp
	$(MOJO) build -I . -O3 tests/test_q4k_mojo_real.mojo $(TP_XLINK) -o tests/test_q4k_mojo_real
	./tests/test_q4k_mojo_real

# Q4_K multi-size performance test - verify stability across different batch sizes
test-q4k-multisize: tp
	$(MOJO) build -I . -O3 tests/test_q4k_multisize.mojo $(TP_XLINK) -o tests/test_q4k_multisize
	./tests/test_q4k_multisize

# Q5_K performance test
test-q5k-perf: tp
	$(MOJO) build -I . -O3 tests/test_q5k_mojo_real.mojo $(TP_XLINK) -o tests/test_q5k_mojo_real
	./tests/test_q5k_mojo_real

# Q5_K multi-size performance test
test-q5k-multisize: tp
	$(MOJO) build -I . -O3 tests/test_q5k_multisize.mojo $(TP_XLINK) -o tests/test_q5k_multisize
	./tests/test_q5k_multisize

# Q6_K performance test
test-q6k-perf: tp
	$(MOJO) build -I . -O3 tests/test_q6k_mojo_real.mojo $(TP_XLINK) -o tests/test_q6k_mojo_real
	./tests/test_q6k_mojo_real

# Q6_K multi-size performance test - Mojo implementation only
test-q6k-multisize: tp
	$(MOJO) build -I . -O3 tests/test_q6k_multisize.mojo $(TP_XLINK) -o tests/test_q6k_multisize
	./tests/test_q6k_multisize

# IQ4_XS × Q8_K performance test (importance quantization)
test-iq4xs-perf: tp
	$(MOJO) build -I . -O3 tests/bench_iq4xs_neon.mojo $(TP_XLINK) -o tests/bench_iq4xs_neon
	./tests/bench_iq4xs_neon

test-iq3s-perf: tp
	$(MOJO) build -I . -O3 tests/bench_iq3s_neon.mojo $(TP_XLINK) -o tests/bench_iq3s_neon
	./tests/bench_iq3s_neon

test-iq3s-correctness: tp
	$(MOJO) build -I . -O3 tests/test_iq3s_correctness.mojo $(TP_XLINK) -o tests/test_iq3s_correctness
	./tests/test_iq3s_correctness

test-iq3xxs-perf: tp
	$(MOJO) build -I . -O3 tests/bench_iq3xxs_neon.mojo $(TP_XLINK) -o tests/bench_iq3xxs_neon
	./tests/bench_iq3xxs_neon

test-iq2s-perf: tp
	$(MOJO) build -I . -O3 tests/bench_iq2s_neon.mojo $(TP_XLINK) -o tests/bench_iq2s_neon
	./tests/bench_iq2s_neon

test-iq2s-perf: tp
		$(MOJO) build -I . -O3 tests/bench_iq2s_neon.mojo $(TP_XLINK) -o tests/bench_iq2s_neon
		./tests/bench_iq2s_neon

test-iq2s-correctness: tp
		$(MOJO) build -I . -O3 tests/test_iq2s_correctness.mojo $(TP_XLINK) -o tests/test_iq2s_correctness
		./tests/test_iq2s_correctness

# IQ series full tests
test-iq-series: test-iq4xs-perf test-iq3s-perf test-iq3xxs-perf test-iq2s-perf

# FP16/FP32 weight-major matmul performance test (actual inference kernel)
test-fp-matmul: tp
	$(MOJO) build -I . -O3 tests/test_fp_weight_matmul.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_fp_weight_matmul
	./tests/test_fp_weight_matmul

# Test MMLA support (requires i8mm extension - ARMv8.6-A+)
# M1: no i8mm, will fail
# M2/M3/M4: has i8mm, should work
test-mmla: tp
	$(MOJO) build --target-features "+i8mm" -I . tests/test_q4k_kernel.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_q4k_kernel_mmla
	./tests/test_q4k_kernel_mmla

# Utility scripts for debugging
dump-gguf:
	pixi run python tools/dump_gguf_meta.py

ref-forward:
	pixi run python tools/ref_forward.py

# Remove every build artifact: the CLI binaries, the tools binaries, the
# compiled shared libraries (C-API + the C runtime helper dylib), the
# generated src/version.mojo, the setuptools egg-info (auto-generated by
# `pip install -e python/`), and all test executables (tests/test_* without
# the .mojo source extension).
clean:
	rm -f it-server it-cli it-rpc-server infer_train
	rm -f tools/check_mem tools/bench_pool bench_cpu bench_transformer_core bench_blas bench_dequant bench_simd
	rm -f bench_q8k_vs_fp32 bench_q8k_sdot_threaded bench_qmatmul_threading
	rm -f tests/bench_qwen3
	rm -f tests/bench_q4k_matmul tests/bench_hunyuan tests/bench_decode_breakdown tests/bench_decode_gpu tests/test_gpu_ffn
	rm -f tests/test_gpu_weight_proj tests/test_tiled_matmul tests/bench_tiled_matmul tests/bench_dynamic_dispatch
	rm -f tests/test_q2k_mojo_real
	rm -f tests/test_q4k_mojo_real tests/test_q4k_multisize
	rm -f tests/test_q5k_mojo_real tests/test_q5k_multisize
	rm -f tests/test_q6k_mojo_real tests/test_q6k_multisize
	rm -f tests/test_fp_weight_matmul
	rm -f tests/bench_iq4xs tests/test_iq4xs tests/bench_iq4xs_neon tests/bench_iq4xs_single tests/bench_iq4xs_multisize tests/test_neon_tbl
	rm -f tests/bench_iq3s_neon tests/bench_iq3xxs_neon tests/test_iq3s_correctness
	rm -f tests/bench_iq2s_neon tests/test_iq2s_correctness tests/test_q3k_mojo_real
	rm -f python/infer_train/_lib/libinfer_train.dylib \
	      python/infer_train/_lib/libinfer_train_tp.dylib \
	      python/infer_train/_lib/libinfer_train_mwq.dylib
	rm -f src/version.mojo
	rm -rf python/*.egg-info python/build
	@for f in tests/test_*; do case "$$f" in *.mojo|*.c) ;; *) rm -f "$$f";; esac; done

# GPU decode benchmark
# GPU decode benchmark
