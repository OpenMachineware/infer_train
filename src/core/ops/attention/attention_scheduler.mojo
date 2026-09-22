# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/attention/attention_scheduler.mojo
#
# Attention scheduler for multi-threaded flash attention.
# Allocates work items (heads) to threads for parallel execution.
#
# Design:
# - Each head is an independent work unit
# - Scheduler creates work items and distributes to threads
# - Supports both single-request (local) and multi-request (server) scenarios

from ...tensor import Tensor
from .kv_cache import KVCacheLayer
from std.utils import StaticTuple
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.sync import Atomic

# Work item: a range of heads to process
struct AttentionWorkItem:
    var req_id: Int
    var head_start: Int
    var head_end: Int
    var kv_split_start: Int  # For long sequence optimization (future)
    var kv_split_end: Int

# Request context: per-request data for parallel processing
struct RequestContext:
    var req_id: Int
    var position: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var scale: Float32
    var cache: Pointer[KVCacheLayer, MutUntrackedOrigin]
    var q_ptr: Pointer[Scalar[DType.float16], MutUntrackedOrigin]
    var out_ptr: Pointer[Scalar[DType.float16], MutUntrackedOrigin]
    var completed_heads: Atomic[Int]

# Scheduler state (shared across threads)
struct SchedulerState:
    var work_items: Pointer[AttentionWorkItem, MutUntrackedOrigin]
    var n_work_items: Atomic[Int]
    var next_item: Atomic[Int]
    var requests: Pointer[RequestContext, MutUntrackedOrigin]
    var n_requests: Int

# Global scheduler state (singleton for worker access)
var _g_scheduler: Pointer[SchedulerState, MutUntrackedOrigin] =
    Pointer[SchedulerState, MutUntrackedOrigin].unsafe_dangling()

# Head-level parallelism: each work item handles HEADS_PER_ITEM heads
comptime HEADS_PER_ITEM = 2  # Tunable: balance between parallelism and overhead


def init_scheduler_state(
    work_items: Pointer[AttentionWorkItem, MutUntrackedOrigin],
    requests: Pointer[RequestContext, MutUntrackedOrigin],
    n_requests: Int,
    n_total_items: Int,
) -> SchedulerState:
    """Initialize scheduler state for a batch of requests."""
    var state = SchedulerState()
    state.work_items = work_items
    state.n_work_items.store(n_total_items)
    state.next_item.store(0)
    state.requests = requests
    state.n_requests = n_requests
    return state


def create_work_items(
    requests: Pointer[RequestContext, MutUntrackedOrigin],
    n_requests: Int,
) -> Tuple[Pointer[AttentionWorkItem, MutUntrackedOrigin], Int]:
    """Create work items for all requests.

    Returns pointer to work items array and total count.
    Caller is responsible for freeing the memory.
    """
    # Count total work items needed
    var total_items = 0
    for r in range(n_requests):
        var req = requests.unsafe_offset(r).unsafe_load()
        var n_heads = req.n_heads
        total_items += (n_heads + HEADS_PER_ITEM - 1) // HEADS_PER_ITEM

    # Allocate work items
    var items = unsafe_alloc[AttentionWorkItem](total_items)

    # Fill work items
    var idx = 0
    for r in range(n_requests):
        var req = requests.unsafe_offset(r).unsafe_load()
        var h = 0
        while h < req.n_heads:
            var item = AttentionWorkItem()
            item.req_id = req.req_id
            item.head_start = h
            item.head_end = min(h + HEADS_PER_ITEM, req.n_heads)
            item.kv_split_start = 0
            item.kv_split_end = 0
            items.unsafe_offset(idx).unsafe_store(val=item)
            idx += 1
            h += HEADS_PER_ITEM

    return (items, total_items)


def get_next_work_item(
    state: Pointer[SchedulerState, MutUntrackedOrigin],
) -> Optional[AttentionWorkItem]:
    """Get next work item from the queue (thread-safe).

    Returns None if all items have been claimed.
    """
    var idx = state.next_item.fetch_add(1)
    if idx < state.n_work_items.load():
        return state.work_items.unsafe_offset(idx).unsafe_load()
    return None


def get_request(
    state: Pointer[SchedulerState, MutUntrackedOrigin],
    req_id: Int,
) -> RequestContext:
    """Get request context by ID."""
    # Linear search for now; can optimize with dictionary later
    for r in range(state.n_requests):
        var req = state.requests.unsafe_offset(r).unsafe_load()
        if req.req_id == req_id:
            return req
    # Should not reach here
    return RequestContext()


# Worker function for parallel head processing
# Note: This is called from the thread pool (C pthread)
@export
abi("C")
def attention_head_worker(
    ctx: Pointer[UInt8, MutUntrackedOrigin],
    idx: Int64,
):
    """Worker function for parallel flash attention.

    Context: SchedulerState pointer
    Each invocation processes one work item (range of heads).
    """
    # Import flash attention kernel
    from .flash_attention_cpu import flash_attention_decode

    # Cast context to scheduler state
    var state = ctx.unsafe_bitcast[SchedulerState]()

    # Process work items until queue is empty
    var item_opt = get_next_work_item(state)
    while item_opt is not None:
        var item = item_opt.value()
        var req = get_request(state, item.req_id)

        # Process each head in this work item
        for h in range(item.head_start, item.head_end):
            var kv_head = h * req.n_kv_heads // req.n_heads

            # Extract Q vector for this head
            var q_vec_ptr = req.q_ptr.unsafe_offset(h * req.head_dim)

            # Create a 1D tensor view for the Q vector
            # Note: We pass the raw pointer; flash_attention_decode expects a Tensor
            # For now, we'll call the internal NEON function directly

            # Get cache reference
            var cache_ref = req.cache.unsafe_load()

            # Call flash attention (single head)
            # TODO: Need to refactor flash_attention_decode to accept pointers
            # For now, mark as completed
            pass

        # Mark heads as completed
        req.completed_heads.fetch_add(item.head_end - item.head_start)

        # Get next work item
        item_opt = get_next_work_item(state)
