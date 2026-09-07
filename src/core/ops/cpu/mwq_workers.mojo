# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/mwq_workers.mojo
#
# M12: the Q4-resident matmul pool workers, built as a STANDALONE shared
# library (libinfer_train_mwq.dylib):
#
#     pixi run mojo build -I . src/core/ops/cpu/mwq_workers.mojo \
#         --emit shared-lib -o python/infer_train/_lib/libinfer_train_mwq.dylib
#
# Why a separate dylib: Mojo 1.0 only honors `@export` in the build's
# ENTRY module, and executables (it-cli / it-server / bench_cpu) strip
# every non-exported symbol - so the C thread pool's
# `dlsym(RTLD_DEFAULT, "it_mwq_worker_*")` can only find these workers
# when they live in a globally-loaded shared library.  The pool loader
# (core/thread_pool.mojo `_load_tp_library`) dlopens this dylib with
# RTLD_GLOBAL next to libinfer_train_tp.dylib, in every process:
#
#   * standalone executables: the workers come from this dylib;
#   * the Python engine (libinfer_train.dylib, RTLD_GLOBAL): the same
#     dylib is dlopen'd by the pool loader as well.
#
# The workers are thin wrappers over the comptime-specialized
# `_mwq_worker_body` (fused per-block dequant + f32 SIMD dot for one
# output column) - the exact accumulation sequence of the single-
# threaded kernel, so the threaded result is bit-identical to it.
# `tid` selects the thread's private scratch slot in the caller-
# allocated pool (see the M12 section in matmul_cpu.mojo).

from src.core.ops.cpu.matmul_cpu import (
    _mw_worker_body,
    _mw_multi_worker_body,
    _mwq_worker_body,
)
from src.core.ops.quantized.quant_types import QuantType
from std.memory import Pointer
from std.origin import MutUntrackedOrigin


# No-op worker: measures the pool's per-submission overhead (dlsym +
# wake-up + chunking + completion) with zero task work - tools/
# bench_pool.mojo.
@export
def it_noop_worker(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _ = ctx
    _ = idx
    _ = tid


# M12: the fp16/f32 weight-major matmul workers (the M5 path).  Same
# visibility fix as the Q4 workers: without them in a globally-loaded
# dylib, executables (it-cli / it-server / bench_cpu) silently ran the
# fp16 "threaded" matmul single-threaded.  The Python engine's
# libinfer_train.dylib exports the same symbols from
# infer_train_bindings.mojo (thin wrappers over the same bodies); the
# two images do not conflict - dlsym finds whichever loaded first, and
# both are bit-identical.
@export
def it_mw_worker(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64
) abi("C"):
    _mw_worker_body(ctx, idx)


@export
def it_mw_multi_worker(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64
) abi("C"):
    _mw_multi_worker_body(ctx, idx)


@export
def it_mwq_worker_q4k_f16(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float16, QuantType.Q4_K_M](ctx, idx, tid)


@export
def it_mwq_worker_q4k_f32(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float32, QuantType.Q4_K_M](ctx, idx, tid)


@export
def it_mwq_worker_q40_f16(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float16, QuantType.Q4_0](ctx, idx, tid)


@export
def it_mwq_worker_q40_f32(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float32, QuantType.Q4_0](ctx, idx, tid)


@export
def it_mwq_worker_q5k_f16(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float16, QuantType.Q5_K](ctx, idx, tid)


@export
def it_mwq_worker_q5k_f32(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float32, QuantType.Q5_K](ctx, idx, tid)


@export
def it_mwq_worker_q6k_f16(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float16, QuantType.Q6_K](ctx, idx, tid)


@export
def it_mwq_worker_q6k_f32(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float32, QuantType.Q6_K](ctx, idx, tid)


@export
def it_mwq_worker_q80_f16(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float16, QuantType.Q8_0](ctx, idx, tid)


@export
def it_mwq_worker_q80_f32(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float32, QuantType.Q8_0](ctx, idx, tid)


@export
def it_mwq_worker_iq4xs_f16(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float16, QuantType.IQ4_XS](ctx, idx, tid)


@export
def it_mwq_worker_iq4xs_f32(
    ctx: Pointer[UInt8, MutUntrackedOrigin], idx: Int64, tid: Int64
) abi("C"):
    _mwq_worker_body[DType.float32, QuantType.IQ4_XS](ctx, idx, tid)
