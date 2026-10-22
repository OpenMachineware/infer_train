// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Debug Q4_K MMA kernel with simpler test

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

    // Test case: 1 row, 1 block (256 elements)
    let m = 1;
    let k = 256;
    let nb = 1;

    // Create simple Q4_K data
    let weights_size = 144;
    let mut weights_data = vec![0u8; weights_size];

    // d = 1.0 (FP16 0x3C00)
    weights_data[0] = 0x00;
    weights_data[1] = 0x3C;
    // dmin = 1.0 (FP16 0x3C00)
    weights_data[2] = 0x00;
    weights_data[3] = 0x3C;

    // scales all zero
    for i in 0..12 {
        weights_data[4 + i] = 0;
    }

    // qs all 0x88
    for i in 0..128 {
        weights_data[16 + i] = 0x88;
    }

    // Input all 1.0
    let input_data = vec![1.0f32; k];

    // Expected output (from CPU simulation):
    // Each qs value = 8 (from 0x88)
    // sc = m = -32 (from scales = 0, minus 32)
    // dl = d * sc = 1 * -32 = -32
    // ml = d * m = 1 * -32 = -32
    // Contribution = (dl * v - ml) * y = (-32 * 8 - (-32)) * 1 = (-256 + 32) = -224
    // 256 elements * -224 = -57344

    println!("Expected output: -57344.0");
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

    let output_buffer =
        device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    // Test each NSG
    for nsg in [8, 16, 32] {
        let kernel_name =
            format!("kernel_mul_mv_q4_K_f32_mma_debug_nsg{}", nsg);
        let function = library.get_function(&kernel_name, None).unwrap();
        let pipeline =
            device.new_compute_pipeline_state_with_function(&function).unwrap();

        // Output buffer needs space for partial sums + final result
        let output_size = (nsg + 1) * 4;
        let output_buffer = device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let queue = device.new_command_queue();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        let args =
            kquant_mv_mma_args { ne00: k as u32, ne01: m as u32, nb01: 144 };
        encoder.set_bytes(
            3,
            std::mem::size_of::<kquant_mv_mma_args>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );

        println!(
            "NSG={}: nb={}, blocks per simdgroup = {}",
            nsg,
            nb,
            (nb + nsg - 1) / nsg
        );
        println!("  Threadgroups: 1, Threads per TG: {}", nsg * 32);

        let thread_groups = MTLSize { width: 1, height: 1, depth: 1 };
        let thread_group_size =
            MTLSize { width: (nsg * 32) as u64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read all partial sums
        let output_ptr = output_buffer.contents() as *const f32;
        let partial_sums: Vec<f32> =
            (0..nsg).map(|i| unsafe { *output_ptr.add(i) }).collect();
        let final_result = unsafe { *output_ptr.add(nsg) };

        println!(
            "  Partial sums: {:?}",
            partial_sums.iter().take(4).collect::<Vec<_>>()
        );
        println!("  Final result: {}", final_result);

        let ratio = final_result / -57344.0;
        println!("  Ratio to expected: {:.1}x", ratio);
        println!();
    }
}

#[repr(C)]
struct kquant_mv_mma_args {
    ne00: u32,
    ne01: u32,
    nb01: u64,
}
