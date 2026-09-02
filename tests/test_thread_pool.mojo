# tests/test_thread_pool.mojo
#
# M9: correctness + load-balancing tests for the Mojo-native work-stealing
# pool (src/core/scheduler/thread_pool.mojo).
#
# What is verified:
#   * the runtime initializes idempotently and reports >= 1 worker;
#   * work-stealing produces correct results (every item processed, with the
#     right value) - including a locally-captured `scale`, which is the exact
#     value the legacy implicit-capture bug (#3483) turned into a random
#     address;
#   * every item is processed exactly once (no misses, no duplicates) via a
#     per-item atomic visit counter;
#   * more than one thread actually runs work at the same time (a concurrency
#     high-water mark), i.e. the pool is not silently running inline on the
#     main thread;
#   * the small-n and single-worker fallbacks stay correct;
#   * the pointer-context helper `parallel_fill` writes through an intact
#     pointer (no Int round-trip).
#
# Build/run (see Makefile):
#   make tp
#   pixi run mojo build -I . tests/test_thread_pool.mojo \
#       -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib \
#       -o tests/test_thread_pool && ./tests/test_thread_pool

from src.core.scheduler import (
    FillContext,
    WorkStealingPool,
    ensure_runtime,
    parallel_fill,
    parallel_fill_ctx,
    run_work_stealing,
    runtime_ready,
    worker_count,
)
from std.atomic import Atomic, Ordering
from std.memory.alloc import unsafe_alloc


def abort():
    from std.os.os import abort as _abort

    _abort()


def check(cond: Bool, label: String):
    if not cond:
        print("FAIL:", label)
        abort()


# Spin iterations per probe item: long enough that concurrently running
# workers reliably overlap, short enough that the whole suite stays fast.
comptime _SPIN = 20000


def test_runtime_init():
    # Idempotent: calling it twice must be a no-op the second time.
    ensure_runtime()
    ensure_runtime()
    check(runtime_ready(), "runtime_ready() is false after ensure_runtime")
    var w = worker_count()
    check(w >= 1, "worker_count() < 1")
    print("  runtime ready, worker_count =", w)


def test_work_stealing_correctness():
    # `scale` is a plain local.  The worker reads it through an explicit
    # `{imm scale}` capture.  Under the legacy implicit-capture bug the value
    # would arrive as a random address; the per-item check below catches that.
    comptime n = 10000
    var scale = 5000
    var results = List[Int](length=n, fill=-1)

    def fill(i: Int) {imm scale, mut results}:
        results[i] = i * scale

    var used = run_work_stealing(fill, n)
    check(used >= 1, "run_work_stealing returned < 1 worker")
    for i in range(n):
        if results[i] != i * scale:
            print("FAIL: results[", i, "] = ", results[i], " want ", i * scale)
            abort()
    print("  correctness OK over", n, "items, workers =", used)


def test_work_stealing_exactly_once():
    # Each item must be claimed exactly once by the steal loop: no item is
    # skipped and none is processed twice.  A per-item atomic visit counter
    # makes both failures visible.
    comptime n = 4096
    var visited = unsafe_alloc[Int64](n)
    for i in range(n):
        visited.unsafe_offset(i).unsafe_store(val=Int64(0))

    def bump(i: Int) {imm visited}:
        _ = Atomic[DType.int64].fetch_add[ordering=Ordering.RELAXED](
            visited.unsafe_offset(i), Int64(1)
        )

    _ = run_work_stealing(bump, n)
    for i in range(n):
        var c = Int(visited.unsafe_offset(i).unsafe_load())
        if c != 1:
            print("FAIL: item", i, "processed", c, "times (want 1)")
            abort()
    visited.unsafe_free()
    print("  exactly-once OK over", n, "items")


def test_work_stealing_all_threads_loaded():
    # Concurrency probe: every item increments a shared "active" counter on
    # entry, records the high-water mark, busy-spins (holding its slot), and
    # decrements on exit.  If the pool ran inline on a single thread the peak
    # would be 1; a healthy pool shows several items in flight at once.
    var workers = worker_count()
    comptime n = 256
    var active = unsafe_alloc[Int64](1)
    active.unsafe_offset(0).unsafe_store(val=Int64(0))
    var peak = unsafe_alloc[Int64](1)
    peak.unsafe_offset(0).unsafe_store(val=Int64(0))

    def probe(i: Int) {imm active, imm peak}:
        var prev = Atomic[DType.int64].fetch_add[
            ordering=Ordering.RELAXED
        ](active, Int64(1))
        var now = Int(prev) + 1
        Atomic[DType.int64].max[ordering=Ordering.RELAXED](
            peak, Int64(now)
        )
        var t = 0
        while t < _SPIN:
            t += 1
        _ = Atomic[DType.int64].fetch_add[ordering=Ordering.RELAXED](
            active, Int64(-1)
        )

    _ = run_work_stealing(probe, n, workers)
    var max_seen = Int(peak.unsafe_offset(0).unsafe_load())
    var leftover = Int(active.unsafe_offset(0).unsafe_load())
    check(leftover == 0, "active counter not drained (leaked work items)")
    print("  concurrency peak =", max_seen, "of", workers, "workers")
    if workers >= 2:
        check(
            max_seen >= 2,
            "only one thread ran work (peak " + String(max_seen) + ")",
        )
    active.unsafe_free()
    peak.unsafe_free()


def test_work_stealing_small_n():
    # Fewer items than workers: the pool must clamp the worker count and
    # still produce correct results.
    comptime n = 3
    var results = List[Int](length=n, fill=-1)

    def fill(i: Int) {mut results}:
        results[i] = i + 100

    var used = run_work_stealing(fill, n, 16)
    check(used >= 1 and used <= n, "worker clamp out of range: " + String(used))
    for i in range(n):
        if results[i] != i + 100:
            print("FAIL: small-n results[", i, "] = ", results[i])
            abort()
    print("  small-n OK, workers =", used)


def test_work_stealing_single_worker():
    # Explicit single-worker dispatch must run inline and stay correct.
    comptime n = 512
    var results = List[Int](length=n, fill=-1)

    def fill(i: Int) {mut results}:
        results[i] = i - 1000

    var used = run_work_stealing(fill, n, 1)
    check(used == 1, "single-worker dispatch used " + String(used))
    for i in range(n):
        if results[i] != i - 1000:
            print("FAIL: single-worker results[", i, "] = ", results[i])
            abort()
    print("  single-worker OK")


def test_parallel_fill_helper():
    # The pointer-context helper: fill a heap buffer through an intact
    # pointer (no Int transcode) and verify every element.
    comptime n = 8192
    var buf = unsafe_alloc[Float32](n)
    var used = parallel_fill(buf, n, Float32(3.5))
    check(used >= 1, "parallel_fill returned < 1 worker")
    for i in range(n):
        if buf.unsafe_offset(i).unsafe_load() != Float32(3.5):
            print("FAIL: parallel_fill buf[", i, "] not set")
            abort()
    buf.unsafe_free()
    print("  parallel_fill OK over", n, "elements, workers =", used)


def test_parallel_fill_ctx():
    # The context-struct pattern (issue #2): the worker reads the buffer
    # pointer, value, and count from a FillContext captured by pointer - it
    # captures no outer local value directly.
    comptime n = 4096
    var buf = unsafe_alloc[Float32](n)
    var ctx = FillContext(buf, Float32(-2.25), n)
    var used = parallel_fill_ctx(ctx)
    check(used >= 1, "parallel_fill_ctx returned < 1 worker")
    for i in range(n):
        if buf.unsafe_offset(i).unsafe_load() != Float32(-2.25):
            print("FAIL: parallel_fill_ctx buf[", i, "] not set")
            abort()
    buf.unsafe_free()
    print("  parallel_fill_ctx OK over", n, "elements, workers =", used)


def test_work_stealing_static_distribution():
    # The `comptime if` branch: oversub = 1 cuts exactly `workers` chunks
    # (one per worker, static split) instead of fine-grained stealing.  Must
    # still process every item exactly once with the right value.
    comptime n = 6000
    var results = List[Int](length=n, fill=-1)
    var scale = 7

    def fill(i: Int) {imm scale, mut results}:
        results[i] = i * scale

    var used = run_work_stealing[oversub=1](fill, n)
    check(used >= 1, "static-distribution returned < 1 worker")
    for i in range(n):
        if results[i] != i * scale:
            print(
                "FAIL: static-distribution results[", i, "] = ", results[i]
            )
            abort()
    print("  static-distribution (comptime if) OK, workers =", used)


def test_pool_handle():
    # The WorkStealingPool handle dispatches onto the same runtime pool and
    # stays correct.
    var pool = WorkStealingPool(0)
    check(pool.workers() >= 1, "pool.workers() < 1")
    comptime n = 2048
    var results = List[Int](length=n, fill=-1)

    def fill(i: Int) {mut results}:
        results[i] = i * 3

    var used = pool.run(fill, n)
    check(used >= 1, "pool.run returned < 1 worker")
    for i in range(n):
        if results[i] != i * 3:
            print("FAIL: pool.run results[", i, "] = ", results[i])
            abort()
    print("  pool handle OK, workers =", used)


def main():
    print("== test_thread_pool ==")
    test_runtime_init()
    test_work_stealing_correctness()
    test_work_stealing_exactly_once()
    test_work_stealing_all_threads_loaded()
    test_work_stealing_small_n()
    test_work_stealing_single_worker()
    test_parallel_fill_helper()
    test_parallel_fill_ctx()
    test_work_stealing_static_distribution()
    test_pool_handle()
    print("test_thread_pool OK")
