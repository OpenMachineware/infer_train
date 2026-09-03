# core/thread_pool.mojo
#
# M5: the engine's thread pool - a thin Mojo wrapper over the C pthread
# task pool in tools/thread_pool.c (libinfer_train_tp.dylib, linked with
# `-Xlinker`).
#
# Mojo 1.0's stdlib ships no threading primitives and the toolchain cannot
# link C object files directly, so the pool is implemented in C and driven
# through `external_call`.  A worker is a Mojo `@export abi("C")` function
# with signature `def worker(ctx: Pointer[UInt8, MutUntrackedOrigin],
# idx: Int64)` - pure pointer math over the caller's context block, no
# allocation and no shared writes, so it is safe on any pool thread.
#
# Usage:
#     var pool = ThreadPool()               # lazily resolves the C entry
#     pool.run("it_mw_worker", ctx, n)      # n tasks over all cores
#     pool.run("it_mw_worker", ctx, n, 4)   # explicit worker count
#
# The shared library is discovered at runtime (dlopen) with RTLD_GLOBAL so
# the C side's dlsym(RTLD_DEFAULT, symbol) finds the Mojo workers - both in
# executables (which dlopen the engine library) and in the engine dylib
# (whose exports are already global when loaded by Python with RTLD_GLOBAL).

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc


def _cstr(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    """Copy a Mojo String into a NUL-terminated buffer for the C side."""
    var bytes = s.as_bytes()
    var n = len(bytes)
    var buf = unsafe_alloc[UInt8](n + 1)
    for i in range(n):
        buf.unsafe_offset(i).unsafe_store(val=bytes[i])
    buf.unsafe_offset(n).unsafe_store(val=UInt8(0))
    return buf


def _load_tp_library():
    """dlopen the C pool library with RTLD_NOW|RTLD_GLOBAL.

    Search order: INFER_TRAIN_TP_LIB env override, then the standard
    locations next to the engine library.
    """
    comptime dlopen = external_call[
        "dlopen",
        Pointer[UInt8, MutUntrackedOrigin],
        Pointer[UInt8, MutUntrackedOrigin],
        Int32,
    ]
    # RTLD_NOW | RTLD_GLOBAL = 2 | 8.  Deliberately list-free: Mojo 1.0
    # shared-library builds miscompile container growth, so the two
    # candidates are dlopen'd directly.
    var handle1 = dlopen(_cstr(String("libinfer_train_tp.dylib")), Int32(10))
    if Int(handle1) != 0:
        return
    var handle2 = dlopen(
        _cstr(String("python/infer_train/_lib/libinfer_train_tp.dylib")),
        Int32(10),
    )
    if Int(handle2) != 0:
        return
    _ = external_call["abort", Int32]()


def _threads_env() -> Optional[String]:
    from std.os import getenv

    return getenv("INFER_TRAIN_THREADS")


def num_cpus() -> Int:
    """Hardware logical CPU count (1 when the C pool is unavailable)."""
    _load_tp_library()
    var n = external_call["tp_num_cpus", Int32]()
    if n < 1:
        return 1
    return Int(n)


def num_pcores() -> Int:
    """Performance-core count (the pool default: E-cores add wake-up
    latency without matmul throughput)."""
    _load_tp_library()
    var n = external_call["tp_num_pcores", Int32]()
    if n < 1:
        return 1
    return Int(n)


def now_ns() -> Int:
    """Wall-clock nanoseconds (C tp_now_ns; monotonic on macOS)."""
    _load_tp_library()
    var t = external_call["tp_now_ns", Int64]()
    return Int(t)


def parallel_run(
    symbol: String,
    ctx: Pointer[UInt8, MutUntrackedOrigin],
    n: Int,
    nthreads: Int = 0,
) -> Int:
    """Run `symbol(ctx, i)` for i in [0, n) on `nthreads` threads.

    Returns the C pool's status (0 = ok).  When nthreads <= 0 the whole
    machine is used; n <= 1 or nthreads == 1 runs inline on the caller.
    """
    if n <= 0:
        return 0
    _load_tp_library()
    var threads = nthreads
    if threads <= 0:
        var env_t = _threads_env()
        if env_t:
            var parsed = 0
            var ok = True
            var env_b = env_t.value().as_bytes()
            for i in range(len(env_b)):
                var c = Int(env_b[i])
                if c >= 48 and c <= 57:
                    parsed = parsed * 10 + (c - 48)
                else:
                    ok = False
            if ok and parsed > 0:
                threads = parsed
        if threads <= 0:
            threads = num_pcores()
    var rc = external_call[
        "tp_run",
        Int32,
        Pointer[UInt8, MutUntrackedOrigin],
        Pointer[UInt8, MutUntrackedOrigin],
        Int64,
        Int32,
    ](_cstr(symbol), ctx, Int64(n), Int32(threads))
    return Int(rc)
