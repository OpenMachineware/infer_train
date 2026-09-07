# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# tools/bench_pool.mojo
#
# M12: measures the C thread pool's per-submission overhead (dlsym +
# wake-up + chunking + completion) with a no-op worker: the fixed cost
# every matmul submission pays on top of the task work.
#
# Build:
#   make tp
#   pixi run mojo build -I . tools/bench_pool.mojo \
#       -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib -o bench_pool
#
# Usage: bench_pool [nthreads]

from src.core.thread_pool import now_ns, parallel_run_tid
from std.memory.alloc import unsafe_alloc
from std.sys import argv


def _time_submissions(n: Int, nthreads: Int) -> Int:
    var ctx = unsafe_alloc[UInt8](64)
    var iters = 2000
    var t0 = now_ns()
    for i in range(iters):
        _ = parallel_run_tid("it_noop_worker", ctx, n, nthreads)
    var t1 = now_ns()
    ctx.unsafe_free()
    return Int((t1 - t0) / iters)


def main():
    var arg_list = List[String]()
    for a in argv():
        arg_list.append(String(a))
    var nthreads = 8
    if len(arg_list) >= 2:
        var b = arg_list[1].as_bytes()
        var parsed = 0
        var ok = True
        for i in range(len(b)):
            var c = Int(b[i])
            if c >= 48 and c <= 57:
                parsed = parsed * 10 + (c - 48)
            else:
                ok = False
        if ok and parsed > 0:
            nthreads = parsed
    # warmup: create the pool threads + prime the dlsym path
    for i in range(200):
        _ = parallel_run_tid("it_noop_worker", unsafe_alloc[UInt8](64), 8, nthreads)
    for n in [1, 8, 256, 1024, 4096]:
        var ns = _time_submissions(n, nthreads)
        print(
            "n=" + String(n) + " threads=" + String(nthreads) + ": "
            + String(ns)
            + " ns/submission"
        )
