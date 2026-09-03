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

.PHONY: test test-m3 test-m4 test-m5 test-m6 test-m7 test-gpu test-gguf-split \
        test-rpc test-thread-pool clean tp server cli rpc-server \
        infer_train version

# The C runtime helper library (thread pool + mmap + clock).
tp:
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
	$(MOJO) build -I . tests/test_registry.mojo $(TP_XLINK) -o tests/test_registry && ./tests/test_registry
	$(MOJO) build -I . tests/test_e2e.mojo $(TP_XLINK) -o tests/test_e2e && ./tests/test_e2e

# M3 tests (tests/ builds with -I . so `src.` imports resolve).
test-m3: tp
	$(MOJO) build -I . tests/test_json.mojo $(TP_XLINK) -o tests/test_json
	./tests/test_json
	$(MOJO) build -I . tests/test_tokenizer.mojo $(TP_XLINK) -o tests/test_tokenizer
	./tests/test_tokenizer
	$(MOJO) build -I . tests/test_ops.mojo $(TP_XLINK) -o tests/test_ops
	./tests/test_ops
	$(MOJO) build -I . tests/test_sampler.mojo $(TP_XLINK) -o tests/test_sampler
	./tests/test_sampler
	$(MOJO) build -I . tests/test_forward.mojo $(TP_XLINK) -o tests/test_forward
	./tests/test_forward
	$(MAKE) test-gguf-split
	$(MAKE) test-gpu

# GPU kernels (Metal).  Falls back to the CPU kernels on machines without a
# Metal GPU, so it is safe to run everywhere.
test-gpu: tp
	$(MOJO) build -I . tests/test_gpuops.mojo $(TP_XLINK) -o tests/test_gpuops
	./tests/test_gpuops
	$(MOJO) build -I . tests/test_gpu_pipeline.mojo $(TP_XLINK) -o tests/test_gpu_pipeline
	./tests/test_gpu_pipeline

# GGUF split-file (multi-part) loading.  Needs the split part files next to
# the repo root (see tests/test_gguf_split.mojo); reports SKIP if absent.
test-gguf-split: tp
	$(MOJO) build -I . tests/test_gguf_split.mojo $(TP_XLINK) -o tests/test_gguf_split
	./tests/test_gguf_split

# M5: the optimizer/CFG/JIT suites (Mojo executables).
test-m5-mojo: tp
	$(MOJO) build -I . tests/test_optimizer.mojo $(TP_XLINK) -o tests/test_optimizer
	./tests/test_optimizer
	$(MOJO) build -I . tests/test_jit.mojo $(TP_XLINK) -o tests/test_jit
	./tests/test_jit

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
	$(MOJO) build -I . tests/test_mmdl.mojo $(TP_XLINK) -o tests/test_mmdl
	./tests/test_mmdl
	$(MOJO) build -I . tests/test_finetune.mojo $(TP_XLINK) -o tests/test_finetune
	./tests/test_finetune
	$(MOJO) build -I . tests/test_kv_cache_m7.mojo $(TP_XLINK) -o tests/test_kv_cache_m7
	./tests/test_kv_cache_m7
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

# M10: the it-server binary (llama-server-compatible HTTP service; the
# renamed M7 CLI - its generation role moved to it-cli, its RPC worker to
# it-rpc-server).  Shared code: src/core/cli_common + src/core/http.
server: tp version
	$(MOJO) build -I . src/core/server-cli/it_server.mojo $(TP_XLINK) -o it-server

# M10: the it-cli binary (llama-cli-compatible quick-verification CLI).
cli: tp version
	$(MOJO) build -I . src/core/cli/it_cli.mojo $(TP_XLINK) -o it-cli

# M8: the it-rpc-server worker binary (llama.cpp-style `llama-rpc-server`),
# split out of the main CLI in M10.
rpc-server: tp version
	$(MOJO) build -I . src/core/server-cli/it_rpc_server.mojo $(TP_XLINK) \
		-o it-rpc-server

# Legacy alias: the old `infer_train` binary name now builds the it-server
# entry (same code, different output name).
infer_train: tp version
	$(MOJO) build -I . src/core/server-cli/it_server.mojo $(TP_XLINK) -o infer_train

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

# Remove every build artifact: the CLI binaries, the compiled shared
# libraries (C-API + the C runtime helper dylib), the generated
# src/version.mojo, the setuptools egg-info (auto-generated by
# `pip install -e python/`), and all test executables (tests/test_* without
# the .mojo source extension).
clean:
	rm -f it-server it-cli it-rpc-server infer_train
	rm -f python/infer_train/_lib/libinfer_train.dylib \
	      python/infer_train/_lib/libinfer_train_tp.dylib
	rm -f src/version.mojo
	rm -rf python/*.egg-info python/build
	@for f in tests/test_*; do case "$$f" in *.mojo) ;; *) rm -f "$$f";; esac; done
