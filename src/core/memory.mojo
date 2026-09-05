# core/memory.mojo
#
# Bulk memory management for the runtime.
#
# The `MemoryPool` grabs one large block up front and hands out aligned
# sub-ranges by bumping an offset.  It deliberately does NOT track individual
# tensor lifetimes: a pool is reset wholesale between inference requests,
# which is the fast-path allocation strategy for a serving runtime.
#
# CPU and GPU pools are distinguished by the `Device` carried in the pool; on
# M1 the GPU pool reuses the same host allocation path because pure Mojo 1.0
# has no host-side GPU allocator (see tensor.mojo for the same note).

from .device import Device
from .utils import align_up, unimplemented
from .thread_pool import _load_tp_library, _cstr
from std.io.file import FileHandle
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.collections import Span

# 1 GiB default pool size, matching the M1 target's unified memory budget.
comptime DEFAULT_POOL_SIZE: Int = 1_073_741_824


def mmap_file(
    file_path: String,
) raises -> Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]:
    """Map (read) a whole file into memory and return (base, size).

    M7: this is now a true `mmap(2)` through the C runtime-helper library
    (`it_mmap` / `it_munmap` in tools/thread_pool.c).  The M2 read-based
    stand-in failed with EINVAL on files > 4 GiB and doubled peak memory
    for the 27B-class models (a 19.7 GB file plus ~51 GiB of fp16 weights);
    a real mapping pages lazily and the OS reclaims clean pages under
    memory pressure.

    The caller must release the mapping with `munmap_file(ptr, size)`.
    """
    _load_tp_library()
    var size_slot = unsafe_alloc[Int64](1)
    size_slot.unsafe_store(val=Int64(0))
    var ptr = external_call[
        "it_mmap",
        Pointer[UInt8, MutUntrackedOrigin],
        Pointer[UInt8, MutUntrackedOrigin],
        Pointer[Int64, MutUntrackedOrigin],
    ](_cstr(file_path), size_slot)
    var size = Int(size_slot.unsafe_load())
    size_slot.unsafe_free()
    if Int(ptr) == 0 or size <= 0:
        unimplemented("mmap_file: cannot map " + file_path)
    return (ptr, size)


def munmap_file(ptr: Pointer[UInt8, MutUntrackedOrigin], size: Int):
    """Release a mapping returned by `mmap_file`."""
    if Int(ptr) == 0 or size <= 0:
        return
    _load_tp_library()
    _ = external_call[
        "it_munmap",
        Int32,
        Pointer[UInt8, MutUntrackedOrigin],
        Int64,
    ](ptr, Int64(size))


def process_rss_bytes() -> Int:
    """The process's current memory footprint in bytes (macOS
    `phys_footprint`; 0 when the C helper is unavailable).

    Q4-resident: with weights kept in their quantized on-disk layout
    (mmap-backed, paged lazily), this measures how much of the model is
    actually resident - the number the `tests/test_gguf.mojo` memory
    budget asserts against.

    NOTE: macOS `phys_footprint` EXCLUDES clean file-backed pages (they
    are reclaimable page cache, not committed memory).  An mmap'd
    quantized model's resident file pages are reported by
    `process_resident_bytes` instead; use it for the total physical RAM
    a fully-touched model occupies.
    """
    _load_tp_library()
    var r = external_call["it_rss_bytes", Int64]()
    if r < 0:
        return 0
    return Int(r)


def process_resident_bytes() -> Int:
    """The process's TOTAL resident bytes (macOS
    `mach_task_basic_info.resident_size`), INCLUDING the mmap'd model
    file's resident pages.

    Q4-resident measurement: after faulting in every mapped page, this
    must be ~= the model's on-disk (quantized) payload size - NOT 2x it.
    The legacy full-dequantize path materializes ~2x the payload as
    anonymous fp16, which shows up in BOTH this number and
    `process_rss_bytes`; the quantized path shows up only here (as
    reclaimable file pages) and keeps the footprint small.
    """
    _load_tp_library()
    var r = external_call["it_resident_bytes", Int64]()
    if r < 0:
        return 0
    return Int(r)


struct MemoryPool:
    var _base: Pointer[UInt8, MutUntrackedOrigin]
    var _size: Int
    var _offset: Int
    var _device: Device

    def __init__(
        out self,
        size: Int = DEFAULT_POOL_SIZE,
        device: Device = Device.CPU,
    ):
        self._size = size
        self._offset = 0
        self._device = device
        var alloc_size = size
        if alloc_size < 1:
            alloc_size = 1
        self._base = unsafe_alloc[UInt8](alloc_size, alignment=64)

    def allocate(
        mut self, size: Int, alignment: Int = 64
    ) -> Pointer[UInt8, MutUntrackedOrigin]:
        """Return a pointer to `size` aligned bytes, bumping the offset.

        Alignment defaults to a 64-byte cache-line friendly value, which also
        satisfies the 16-byte alignment required by NEON/SSE and the larger
        alignments expected by the Metal/NVIDIA backends.
        """
        var aligned = align_up(self._offset, alignment)
        var end = aligned + size
        if end > self._size:
            abort_oob()
        self._offset = end
        return self._base.unsafe_offset(aligned)

    def reset(mut self):
        """Invalidate every outstanding allocation (rewind to the start)."""
        self._offset = 0

    def used(self) -> Int:
        return self._offset

    def capacity(self) -> Int:
        return self._size

    def device(self) -> Device:
        return self._device

    def mmap_load(
        self, file_path: String
    ) raises -> Pointer[UInt8, MutUntrackedOrigin]:
        """Map a model file into memory.

        Delegates to the module-level `mmap_file` (see its M2 note); the
        returned buffer is *not* pool-managed and must be freed explicitly.
        """
        var buffer, _ = mmap_file(file_path)
        return buffer


def abort_oob():
    """Abort when a pool allocation overruns its backing buffer."""
    unimplemented("memory pool exhausted")
