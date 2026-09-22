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
        test-iq4xs-perf bench-kv bench-flash bench-batch-attention

mwq:
	$(MOJO) build -I . src/core/ops/cpu/mwq_workers.mojo \
		--emit shared-lib -o $(MWQ)

tp: mwq
	cc -O2 -shared -o $(TP) tools/thread_pool.c

version:
	pixi run python tools/gen_version.py

test: tp
	$(MOJO) build -I . tests/test_core.mojo $(TP_XLINK) -o tests/test_core && ./tests/test_core
	$(MOJO) build -I . tests/test_quant.mojo $(TP_XLINK) -o tests/test_quant && ./tests/test_quant
	$(MOJO) build -I . tests/test_cpuops.mojo $(TP_XLINK) -o tests/test_cpuops && ./tests/test_cpuops
	$(MOJO) build -I . tests/test_registry.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_registry && ./tests/test_registry
	$(MOJO) build -I . tests/test_e2e.mojo $(TP_XLINK) $(BLAS_XLINK) -o tests/test_e2e && ./tests/test_e2e

bench_cpu: tp mwq
	$(MOJO) build -I . tools/bench_cpu.mojo $(TP_XLINK) -Xlinker $(MWQ) $(BLAS_XLINK) -o bench_cpu

clean:
	rm -f it-server it-cli it-rpc-server infer_train
	rm -f tools/check_mem tools/bench_pool bench_cpu bench_transformer_core bench_blas bench_dequant bench_simd
	rm -f python/infer_train/_lib/libinfer_train.dylib \
	      python/infer_train/_lib/libinfer_train_tp.dylib \
	      python/infer_train/_lib/libinfer_train_mwq.dylib
	rm -f src/version.mojo
