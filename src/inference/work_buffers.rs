// SPDX-License-Identifier: AGPL-3.0-or-later
//
// CPU Work Buffers - Pre-allocated buffers for GEMV operations
//
// This module provides pre-allocated buffers that are reused across
// multiple GEMV calls, eliminating allocation overhead.

use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    /// Pre-allocated output buffer for GEMV
    /// Key = (in_size, out_size), Value = output buffer
    static OUTPUT_BUFFERS: RefCell<Vec<(usize, usize, Vec<f32>)>> = RefCell::new(Vec::new());

    /// Pre-allocated Q8_K buffer for quantized input
    static Q8K_BUFFER: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

/// Get or create an output buffer for GEMV
pub fn get_output_buffer(in_size: usize, out_size: usize) -> Vec<f32> {
    OUTPUT_BUFFERS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();

        // Try to find a matching buffer
        for (stored_in, stored_out, buffer) in buffers.iter_mut() {
            if *stored_in == in_size && *stored_out == out_size {
                // Clear and return
                for elem in buffer.iter_mut() {
                    *elem = 0.0;
                }
                return buffer.clone();
            }
        }

        // Create new buffer and cache it
        let buffer = vec![0.0f32; out_size];
        buffers.push((in_size, out_size, buffer.clone()));
        buffer
    })
}

/// Get the Q8_K buffer for a given size
/// Returns a mutable reference that can be used for quantization
pub fn get_q8k_buffer(size: usize) -> Vec<u8> {
    Q8K_BUFFER.with(|buffer_cell| {
        let mut buffer = buffer_cell.borrow_mut();
        if buffer.len() < size {
            buffer.resize(size, 0);
        }
        buffer.clone()
    })
}

/// Clear all cached buffers
pub fn clear_buffers() {
    OUTPUT_BUFFERS.with(|buffers| {
        buffers.borrow_mut().clear();
    });
    Q8K_BUFFER.with(|buffer| {
        buffer.borrow_mut().clear();
    });
}
