// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Metal device layer - shared resources across all backend contexts
//!
//! This follows llama.cpp's design where CommandQueue and Library are
//! shared at the device level, not recreated for each context.

use metal::{CommandQueue, Device, MTLResourceOptions};
use std::sync::OnceLock;

/// Shared Metal device resources (singleton)
///
/// This is the "device layer" in llama.cpp's two-layer architecture.
/// All `MetalContext` instances share this one `MetalDevice`.
///
/// # Why this design?
///
/// 1. Metal device (`MTLCreateSystemDefaultDevice`) is a singleton
/// 2. Creating multiple `CommandQueue`s on the same device
///    can corrupt Metal's internal state
/// 3. llama.cpp explicitly shares queue at device level:
///    "a single global queue shared by all Metal backends"
pub struct MetalDevice {
    /// Metal device (singleton from MTLCreateSystemDefaultDevice)
    device: Device,

    /// Shared command queue - "a single global queue shared by all Metal backends"
    queue: CommandQueue,
}

impl MetalDevice {
    /// Get the global MetalDevice instance
    ///
    /// Creates the device and queue on first access.
    /// All subsequent calls return the same instance.
    pub fn global() -> &'static MetalDevice {
        static INSTANCE: OnceLock<MetalDevice> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            let device =
                Device::system_default().expect("No Metal device found");

            let queue = device.new_command_queue();

            MetalDevice { device, queue }
        })
    }

    /// Get the Metal device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the shared command queue
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Create a new buffer with data
    pub fn new_buffer_with_data(&self, data: &[u8]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a new zeroed buffer
    pub fn new_buffer(&self, size: usize) -> metal::Buffer {
        let buffer = self
            .device
            .new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        // Zero the buffer
        unsafe {
            std::ptr::write_bytes(buffer.contents() as *mut u8, 0, size);
        }
        buffer
    }
}
