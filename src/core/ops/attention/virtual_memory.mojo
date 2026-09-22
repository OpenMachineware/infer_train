# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# Virtual memory management for zero-overhead Paged KV Cache
#
# Key insight: Reserve virtual address space, commit on demand.
# This makes Paged KV access as fast as Dense (no block_table lookup).

# C library functions via extern
@extern
def mmap(
    addr: UnsafePointer[UInt8, MutUntrackedOrigin],
    len: Int,
    prot: Int,
    flags: Int,
    fd: Int,
    offset: Int
) -> UnsafePointer[UInt8, MutUntrackedOrigin]

@extern
def mprotect(addr: UnsafePointer[UInt8, MutUntrackedOrigin], len: Int, prot: Int) -> Int

@extern
def munmap(addr: UnsafePointer[UInt8, MutUntrackedOrigin], len: Int) -> Int

# Constants
comptime PROT_NONE = 0x0
comptime PROT_READ = 0x1
comptime PROT_WRITE = 0x2
comptime MAP_PRIVATE = 0x2
comptime MAP_ANONYMOUS = 0x1000  # macOS
comptime MAP_NORESERVE = 0x40

struct VirtualMemoryRegion:
    """A region of virtual address space that can grow on demand."""

    var base: UnsafePointer[UInt8, MutUntrackedOrigin]
    var reserved_bytes: Int
    var committed_bytes: Int
    var page_size: Int

    def __init__(out self, reserve_bytes: Int, page_size: Int = 4096):
        """Reserve virtual address space (no physical memory)."""
        self.reserved_bytes = reserve_bytes
        self.committed_bytes = 0
        self.page_size = page_size

        self.base = mmap(
            None,  # Let OS choose address
            reserve_bytes,
            PROT_NONE,  # No access until committed
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
            -1,
            0
        )

        if self.base:
            raise Error("mmap failed")

    def commit(mut self, offset: Int, size: Int):
        """Commit physical memory for a range."""
        if offset + size > self.reserved_bytes:
            raise Error("Commit beyond reserved range")

        var start = (offset // self.page_size) * self.page_size
        var commit_size = ((offset + size + self.page_size - 1) // self.page_size) * self.page_size - start

        if mprotect(self.base.unsafe_offset(start), commit_size, PROT_READ | PROT_WRITE) != 0:
            raise Error("mprotect failed")

        if start + commit_size > self.committed_bytes:
            self.committed_bytes = start + commit_size

    def ensure_capacity(mut self, needed_bytes: Int):
        """Ensure enough memory is committed."""
        if needed_bytes > self.committed_bytes:
            self.commit(self.committed_bytes, needed_bytes - self.committed_bytes)

    def data(self) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
        return self.base

    def destroy(mut self):
        if self.base:
            munmap(self.base, self.reserved_bytes)
            self.base = None
