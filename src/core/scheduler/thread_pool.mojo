# core/scheduler/thread_pool.mojo
#
# A Mojo-native work-stealing thread pool for the engine's CPU kernels.
#
# Why this exists
# ---------------
# The previous pool (core/thread_pool.mojo) is a thin wrapper over a C
# pthread library.  It works, but several Mojo-side call sites that used
# `parallelize` directly exhibited the classic "only the main thread works,
# worker threads sit at 0% CPU" failure.  Root causes, all fixed here:
#
#   1. Missing runtime initialization.  When the engine is built as a shared
#      library (`mojo build --emit shared-lib`) and driven from a non-Mojo
#      host (Python / C), no Mojo `main()` runs, so the async runtime - the
#      thread pool that `parallelize` / `sync_parallelize` depend on - is
#      never initialized.  `parallelism_level()` then reports 1 and every
#      "parallel" call silently runs inline on the caller.  Fix: call
#      `initialize_runtime()` (idempotent, cheap) before any parallel API.
#      `ensure_runtime()` below is the single entry point for that.
#
#   2. Local-value capture garbage.  Capturing a plain local in a worker via
#      the legacy implicit-capture / `@parameter` style can yield a random
#      address instead of the value (Mojo issue #3483).  Fix: never rely on
#      implicit capture.  Every worker closure uses an *explicit* capture
#      list (`{imm ...}` / `{mut ...}` / `{var ...}`), and large payloads are
#      passed by pointer, not by value.
#
#   3. Pointer-to-Int round-trips.  `Int64(Int(ptr))` strips the origin, so
#      the compiler may free the buffer before the worker reads it (use-after-
#      free).  Fix: keep `Pointer` values whole; never transcode a pointer
#      through `Int`.
#
# Work stealing
# -------------
# `run_work_stealing` dispatches `workers` concurrent tasks to the runtime's
# thread pool.  All tasks draw from a single shared atomic counter - the
# "victim queue".  Each task claims the next chunk of work items, does the
# work, and claims again until the queue is drained.  Fast threads therefore
# claim more chunks and slow threads fewer: the load self-balances, which is
# the work-stealing property (a thread that finishes early steals the next
# chunk instead of idling).  This is the dynamic-scheduling analogue of a
# per-thread deque and is the strongest load balancing Mojo's range-parallel
# API can express without hand-rolled threads.
#
# Usage
# -----
#     from src.core.scheduler import ensure_runtime, run_work_stealing
#
#     ensure_runtime()                       # FFI entry points
#     var out = List[Int](length=n, fill=0)
#     def fill(i: Int) {mut out}:
#         out[i] = i * 2
#     var used = run_work_stealing(fill, n)  # n items, all cores
#
# The pool is synchronous: `run_work_stealing` returns only after every item
# has been processed, so captured stack state (lists, pointers) is guaranteed
# to be alive for the whole call.

from max.algorithm import sync_parallelize
from std.atomic import Atomic, Ordering
from std.math import ceildiv
from std.memory.alloc import unsafe_alloc
from std.runtime import initialize_runtime
from std.runtime.asyncrt import parallelism_level


# ===-----------------------------------------------------------------------===#
# Runtime initialization (the FFI fix)
# ===-----------------------------------------------------------------------===#


def ensure_runtime():
    """Idempotently initialize the Mojo async runtime (its thread pool).

    Must run before any parallel / asynchronous API when the engine is loaded
    as a shared library by a non-Mojo host, where no Mojo `main()` starts the
    runtime.  Cheap and a no-op when the runtime is already up; one call
    covers every thread in the process.  Safe to call at the top of every
    `@export` entry point and at the top of every interpreter run.
    """
    initialize_runtime()


def worker_count() -> Int:
    """The runtime thread-pool size (the pool's default worker count).

    Always at least 1.  Reads the runtime's real parallelism level, which is
    only meaningful after `ensure_runtime()` has run.
    """
    ensure_runtime()
    var n = parallelism_level()
    if n < 1:
        return 1
    return n


def runtime_ready() -> Bool:
    """True once the runtime reports at least one worker thread."""
    ensure_runtime()
    return parallelism_level() >= 1


# ===-----------------------------------------------------------------------===#
# Work-stealing dispatch
# ===-----------------------------------------------------------------------===#

# How many more chunks than workers to cut the work into.  More chunks than
# threads gives the steal loop room to rebalance; 8x keeps each chunk a
# meaningful amount of work so the atomic claim cost stays amortized.
comptime _OVERSUBSCRIPTION = 8


def run_work_stealing[
    FuncType: def(Int) -> None,
    oversub: Int = _OVERSUBSCRIPTION,
](func: FuncType, n_work_items: Int, nworkers: Int = 0) -> Int:
    """Run `func(0) ... func(n_work_items - 1)` with dynamic load balancing.

    The items are cut into chunks; `workers` concurrent tasks claim chunks
    from a shared atomic counter until the queue is drained (work stealing).
    Returns the number of worker tasks actually dispatched (>= 1 when
    `n_work_items > 0`).

    `func` must be safe to run concurrently on distinct item ranges and must
    not allocate large stack objects (worker stacks are small).  All state it
    needs should arrive through explicit captures or pointers, never through
    implicit local capture.

    Parameters:
        oversub: Compile-time chunk oversubscription factor.  `> 1` cuts
            `workers * oversub` chunks (fine-grained stealing); `<= 1` cuts
            exactly `workers` chunks (one per worker, static distribution).
            Selected with `comptime if` - the modern replacement for the
            removed `@parameter if`.

    Args:
        func: The per-item body, `def(Int) -> None`.
        n_work_items: Number of items to process.  <= 0 is a no-op.
        nworkers: Worker count.  <= 0 means "use the whole runtime pool".
    """
    ensure_runtime()
    if n_work_items <= 0:
        return 0

    var workers = nworkers
    if workers <= 0:
        workers = worker_count()
    if workers < 1:
        workers = 1
    if n_work_items == 1:
        # A single item: run inline on the caller.
        func(0)
        return 1
    if workers == 1:
        # Single worker: process every item inline (serial), not just the
        # first - the pool round-trip would buy nothing here.
        for i in range(n_work_items):
            func(i)
        return 1
    if workers > n_work_items:
        workers = n_work_items

    # Shared work queue: one atomic counter the workers steal from.  Heap
    # allocated so the address stays valid for the (synchronous) call and the
    # pointer keeps its origin (no Int round-trip).
    var counter = unsafe_alloc[Int64](1)
    counter.unsafe_offset(0).unsafe_store(val=Int64(0))

    # Compile-time branch on the oversubscription factor (the `comptime if`
    # form, not the removed `@parameter if`): fine-grained chunks for
    # stealing, or one chunk per worker for a static split.  `num_chunks` is
    # declared outside the branch so it is visible after it.
    var num_chunks = workers
    comptime if oversub > 1:
        num_chunks = workers * oversub
    if num_chunks > n_work_items:
        num_chunks = n_work_items
    var chunk_size = ceildiv(n_work_items, num_chunks)
    if chunk_size < 1:
        chunk_size = 1

    # Explicit capture list: `func` by immutable reference (it is a value
    # that outlives the synchronous call), the counter pointer and the two
    # sizes by value.  No implicit capture of any local.
    def steal(task_id: Int) {imm func, imm counter, imm n_work_items, imm chunk_size}:
        while True:
            # Claim the next chunk id.  `fetch_add` returns the prior value,
            # so this is an atomic "take the next slot" over the shared queue.
            var prev = Atomic[DType.int64].fetch_add[
                ordering=Ordering.RELAXED
            ](counter, Int64(1))
            var start = Int(prev) * chunk_size
            if start >= n_work_items:
                break
            var end = start + chunk_size
            if end > n_work_items:
                end = n_work_items
            for i in range(start, end):
                func(i)

    sync_parallelize(steal, workers)
    counter.unsafe_free()
    return workers


# ===-----------------------------------------------------------------------===#
# Pool handle
# ===-----------------------------------------------------------------------===#


struct WorkStealingPool(Movable):
    """A named work-stealing pool pinned to a worker count.

    `nworkers = 0` tracks the runtime pool size at construction.  The pool is
    stateless apart from that count: the runtime's thread pool is shared and
    process-wide, so every pool instance dispatches onto the same workers.
    """

    var nworkers: Int

    def __init__(out self, nworkers: Int = 0):
        ensure_runtime()
        self.nworkers = nworkers

    def workers(self) -> Int:
        """The effective worker count (>= 1)."""
        if self.nworkers <= 0:
            return worker_count()
        return self.nworkers

    def run[FuncType: def(Int) -> None](
        self, func: FuncType, n_work_items: Int
    ) -> Int:
        """Dispatch `n_work_items` items of `func` onto the pool."""
        return run_work_stealing(func, n_work_items, self.workers())


# ===-----------------------------------------------------------------------===#
# Concrete helpers (pointer-context pattern)
# ===-----------------------------------------------------------------------===#


def parallel_fill(
    buf: Pointer[Float32, MutUntrackedOrigin],
    n: Int,
    value: Float32,
    nworkers: Int = 0,
) -> Int:
    """Fill `buf[0..n)` with `value` on the work-stealing pool.

    Demonstrates the pointer-context pattern the kernels should follow: the
    worker reads the buffer pointer and the fill value from explicit captures
    - the pointer keeps its origin (no `Int` transcode) and no local value is
    captured implicitly.
    """
    def fill(i: Int) {imm buf, imm value}:
        buf.unsafe_offset(i).unsafe_store(val=value)

    return run_work_stealing(fill, n, nworkers)


# A bundle of everything a parallel worker needs, passed by pointer so the
# worker reads fields from the struct instead of capturing outer locals
# directly (the issue-#2 fix).  The struct is small and register-passable;
# the large payload it points to (`buf`) lives on the heap.
struct FillContext(Movable, ImplicitlyCopyable):
    var buf: Pointer[Float32, MutUntrackedOrigin]
    var value: Float32
    var n: Int

    def __init__(
        out self,
        buf: Pointer[Float32, MutUntrackedOrigin],
        value: Float32,
        n: Int,
    ):
        self.buf = buf
        self.value = value
        self.n = n


def parallel_fill_ctx(ctx: FillContext, nworkers: Int = 0) -> Int:
    """Fill `ctx.buf[0..ctx.n)` with `ctx.value` using the pool.

    The worker reads the buffer pointer, fill value, and count from the
    context struct (captured by pointer) - it captures no outer local value,
    which is the pattern that avoids the legacy capture-garbage bug.
    """
    def fill(i: Int) {imm ctx}:
        ctx.buf.unsafe_offset(i).unsafe_store(val=ctx.value)

    return run_work_stealing(fill, ctx.n, nworkers)
