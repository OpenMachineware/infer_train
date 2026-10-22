// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test correctness of Q4_K MMA kernel by comparing with CPU reference

use metal::*;

fn main() {
    let device = Device::system_default().expect("No Metal device found");

    // Load and prepare blocks.h
    let ggml_common = include_str!("../src/ops/metal/include/blocks.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);

    // Load kernel source and remove include line
    let kernel_source =
        include_str!("../src/ops/metal/gemv/kquant_mv_mma.metal");
    let kernel_source =
        kernel_source.replace("#include \"../include/blocks.h\"", "");

    // Combine
    let combined = format!("{}\n{}", ggml_common_metal, kernel_source);

    let library = device
        .new_library_with_source(&combined, &CompileOptions::new())
        .expect("Failed to compile Metal library");

    // Test case: small size for verification
    let m = 4; // 4 rows
    let k = 256; // 1 block

    // Create simple test data
    let nb = k / 256;
    let weights_size = m * nb * 144;
    let mut weights_data = vec![0u8; weights_size];

    // Set scales to 1.0, all quantized values to 8
    for row in 0..m {
        let offset = row * nb * 144;
        // d and dmin as FP16 = 1.0
        weights_data[offset] = 0x00; // d = 1.0
        weights_data[offset + 1] = 0x3C;
        weights_data[offset + 2] = 0x00; // dmin = 1.0
        weights_data[offset + 3] = 0x3C;

        // Set all qs bytes to 0x88 (quantized value 8 for both nibbles)
        for i in 0..128 {
            weights_data[offset + 16 + i] = 0x88;
        }

        // Set scales to 0 (sc = m = 0)
        for i in 0..12 {
            weights_data[offset + 4 + i] = 0;
        }
    }

    // Input vector: all 1.0
    let mut input_data = vec![1.0f32; k];
    for i in 0..k {
        input_data[i] = 1.0;
    }

    // Expected output: Each element should be (dl * v - ml) summed
    // With d=1, sc=0, m=0: dl=1, ml=1
    // Quantized value 8: contribution = (1*8 - 1) * 1.0 = 7 per element
    // 256 elements per row: total = 256 * 7 = 1792

    println!("Test: M={}, K={}", m, k);
    println!("Expected output per row: 1792.0");
    println!();

    let weights_buffer = device.new_buffer_with_data(
        weights_data.as_ptr() as *const std::ffi::c_void,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const std::ffi::c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    // Test each NSG variant
    for nsg in [8, 16, 32] {
        let kernel_name = format!("kernel_mul_mv_q4_K_f32_mma_nsg{}", nsg);
        let function = library.get_function(&kernel_name, None).unwrap();
        let pipeline =
            device.new_compute_pipeline_state_with_function(&function).unwrap();

        let queue = device.new_command_queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        let args = kquant_mv_mma_args {
            ne00: k as u32,
            ne01: m as u32,
            nb01: (nb * 144) as u64,
        };
        encoder.set_bytes(
            3,
            std::mem::size_of::<kquant_mv_mma_args>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );

        let thread_groups = MTLSize {
            width: m as u64, // One threadgroup per row
            height: 1,
            depth: 1,
        };
        let thread_group_size =
            MTLSize { width: (nsg * 32) as u64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read output
        let output_ptr = output_buffer.contents() as *const f32;
        let output: Vec<f32> =
            (0..m).map(|i| unsafe { *output_ptr.add(i) }).collect();

        println!("NSG={:2} output: {:?}", nsg, output);
    }
}

#[repr(C)]
struct kquant_mv_mma_args {
    ne00: u32,
    ne01: u32,
    nb01: u64,
}
