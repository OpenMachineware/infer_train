# core/scheduler/__init__.mojo
#
# CPU scheduling: a Mojo-native work-stealing thread pool built on the
# runtime's `parallelize` / `sync_parallelize` (see thread_pool.mojo).
#
# This replaces the "only the main thread works" failure mode of the old
# C-pthread wrapper for the paths that can be expressed as a parallel
# range: the pool guarantees the runtime is initialized (the FFI fix),
# passes all worker data through explicit captures / pointers (no local
# value capture, no `Int` pointer round-trips), and distributes work
# dynamically so every worker thread stays loaded.

from .thread_pool import (
    FillContext,
    WorkStealingPool,
    ensure_runtime,
    parallel_fill,
    parallel_fill_ctx,
    run_work_stealing,
    runtime_ready,
    worker_count,
)
