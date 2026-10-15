// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Debug test for Metal context

fn main() {
    println!("=== Metal Context Debug Test ===\n");

    match infer_train::quant::vec_dot::gpu::metal::MetalContext::new() {
        Ok(ctx) => {
            println!("✓ Metal context created successfully");
            println!("  Device: {}", ctx.device().name());
        }
        Err(e) => {
            println!("✗ Failed to create Metal context: {}", e);
        }
    }
}
