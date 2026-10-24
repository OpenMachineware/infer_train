// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Memory range tracking for concurrent kernel execution
//!
//! Based on llama.cpp's ggml_mem_ranges implementation.
//! Tracks memory access patterns to determine if operations can execute concurrently.

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAccessType {
    /// Read access (source tensor)
    Src,
    /// Write access (destination tensor)
    Dst,
}

/// Represents a memory range in a GPU buffer
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
#[derive(Debug, Clone)]
pub struct MemoryRange {
    /// Buffer pointer as unique ID
    pub buffer_id: usize,
    /// Start address offset
    pub start: usize,
    /// End address offset
    pub end: usize,
    /// Access type (read or write)
    pub access_type: MemoryAccessType,
}

/// Tracks memory ranges to determine concurrent execution safety
///
/// Two operations can run concurrently if:
/// - New src doesn't overlap with existing dst (no read-write conflict)
/// - New dst doesn't overlap with any existing range (no write-write or write-read conflict)
/// - Src-src overlap is allowed (multiple reads of same memory)
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct MemoryRanges {
    ranges: Vec<MemoryRange>,
    debug: bool,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl MemoryRanges {
    pub fn new() -> Self {
        Self {
            ranges: Vec::with_capacity(256),
            debug: std::env::var("INFER_DEBUG_MEM_RANGES").is_ok(),
        }
    }

    /// Clear all tracked ranges
    pub fn reset(&mut self) {
        self.ranges.clear();
    }

    /// Add a memory range from a tensor buffer
    pub fn add(
        &mut self,
        buffer: &Rc<MetalBuffer>,
        offset: usize,
        size: usize,
        access_type: MemoryAccessType,
    ) {
        let buffer_id = buffer.contents() as usize;
        let range = MemoryRange {
            buffer_id,
            start: offset,
            end: offset + size,
            access_type,
        };

        if self.debug {
            eprintln!(
                "[MEM] Add {:?} range: buffer={:x}, [{}, {})",
                access_type, buffer_id, range.start, range.end
            );
        }

        self.ranges.push(range);
    }

    /// Check if a new memory range can be accessed concurrently
    ///
    /// Returns false if:
    /// - New src overlaps with existing dst (read-write conflict)
    /// - New dst overlaps with any existing range (write-write or write-read conflict)
    pub fn check(
        &self,
        buffer: &Rc<MetalBuffer>,
        offset: usize,
        size: usize,
        access_type: MemoryAccessType,
    ) -> bool {
        let buffer_id = buffer.contents() as usize;
        let new_start = offset;
        let new_end = offset + size;

        for range in &self.ranges {
            // Different buffers never conflict
            if range.buffer_id != buffer_id {
                continue;
            }

            // Src-src overlap is allowed (multiple reads)
            if access_type == MemoryAccessType::Src
                && range.access_type == MemoryAccessType::Src
            {
                continue;
            }

            // Check for overlap: [start, end) intersects [new_start, new_end)
            if new_start < range.end && new_end >= range.start {
                if self.debug {
                    eprintln!(
                        "[MEM] Conflict: new {:?} [{}, {}) overlaps with existing {:?} [{}, {})",
                        access_type,
                        new_start,
                        new_end,
                        range.access_type,
                        range.start,
                        range.end
                    );
                }
                return false;
            }
        }

        true
    }

    /// Add all tensor ranges for an operation
    /// Inputs are src, outputs are dst
    pub fn add_tensor(
        &mut self,
        buffer: &Rc<MetalBuffer>,
        size: usize,
        is_input: bool,
    ) {
        let access_type = if is_input {
            MemoryAccessType::Src
        } else {
            MemoryAccessType::Dst
        };
        self.add(buffer, 0, size, access_type);
    }

    /// Check all tensor ranges for an operation
    /// Returns true if the operation can execute concurrently with previous operations
    pub fn check_tensor(
        &self,
        buffer: &Rc<MetalBuffer>,
        size: usize,
        is_input: bool,
    ) -> bool {
        let access_type = if is_input {
            MemoryAccessType::Src
        } else {
            MemoryAccessType::Dst
        };
        self.check(buffer, 0, size, access_type)
    }

    /// Get number of tracked ranges
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Insert GPU memory barrier to ensure proper synchronization
    ///
    /// This is critical for concurrent kernel execution to work correctly.
    /// Based on llama.cpp's use of `memoryBarrierWithScope:`.
    ///
    /// # Safety
    /// Uses Objective-C runtime to call Metal API directly.
    pub fn insert_gpu_barrier(encoder: &ComputeCommandEncoderRef) {
        // Use Objective-C msg_send to call memoryBarrierWithScope:
        // This is the same approach llama.cpp uses
        //
        // Rust metal crate doesn't expose this API directly, so we use
        // the Objective-C runtime.
        //
        // MTLMemoryScopeDevice = 1 (device-wide synchronization)
        #[cfg(target_os = "macos")]
        unsafe {
            use objc::msg_send;
            use objc::runtime::Object;

            let encoder_ptr = encoder.as_ptr() as *mut Object;
            let _: () = msg_send![encoder_ptr, memoryBarrierWithScope: 1usize];
        }

        // Alternative: use memory_barrier_with_resources if available
        // But this requires enumerating all conflicting buffers
        // For simplicity, use device-wide barrier
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl Default for MemoryRanges {
    fn default() -> Self {
        Self::new()
    }
}
