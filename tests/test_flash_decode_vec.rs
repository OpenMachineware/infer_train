// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test compilation of new flash decode vec kernel
// This is a minimal test to verify the kernel compiles

use metal::*;

fn main() {
    let device = Device::system_default().unwrap();

    // Load the new kernel
    let source = include_str!("../src/ops/metal/flash_decode_vec.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);

    match device.new_library_with_source(source, &options) {
        Ok(library) => {
            println!("✓ Kernel compilation successful");

            // Check if our functions exist
            match library.get_function("flash_decode_vec_q4_0", None) {
                Ok(_) => println!("✓ flash_decode_vec_q4_0 function found"),
                Err(e) => println!("✗ flash_decode_vec_q4_0 not found: {}", e),
            }

            match library.get_function("flash_decode_vec_f16", None) {
                Ok(_) => println!("✓ flash_decode_vec_f16 function found"),
                Err(e) => println!("✗ flash_decode_vec_f16 not found: {}", e),
            }
        }
        Err(e) => {
            println!("✗ Kernel compilation failed:");
            println!("{}", e);
        }
    }
}
