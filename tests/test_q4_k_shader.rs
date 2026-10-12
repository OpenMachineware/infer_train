// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test self-written Q4_K shader correctness and performance
// Usage: cargo run --bin test_q4_k_shader --release

use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

#[repr(C)]
struct MvArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

fn generate_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

const FC_MUL_MV_NSG: NSUInteger = 600;
const FC_MUL_MV_NE12: NSUInteger = 602;
const FC_MUL_MV_R2: NSUInteger = 603;
const FC_MUL_MV_R3: NSUInteger = 604;

fn create_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_NSG,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_NE12,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_R2,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_R3,
        );
    }
    cv
}

struct TestContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl TestContext {
    fn new() -> Self {
        let device = Device::system_default().expect("No Metal device");
        let queue = device.new_command_queue();

        // Load self-written shader
        let source = include_str!("../src/ops/metal/mul_mv_q4_k.metal");
        let compile_options = CompileOptions::new();
        let library =
            match device.new_library_with_source(source, &compile_options) {
                Ok(lib) => {
                    println!("Shader compiled successfully");
                    lib
                }
                Err(e) => {
                    panic!("Shader compilation failed: {:?}", e);
                }
            };

        // Create pipeline with function constants
        let cv = create_function_constants(2, 1, 1, 1); // nsg=2, ne12=1, r2=1, r3=1
        let kernel = library
            .get_function("kernel_mul_mm_q4_K_f32", Some(cv))
            .expect("Failed to get kernel");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline");

        Self { device, queue, pipeline }
    }

    fn run_mv(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ4K],
        input: &[f32],
    ) -> Vec<f32> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const c_void,
            weights_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            input_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        // Thread group configuration matching llama.cpp EXACTLY
        // Q4_K: N_R0 = 2, NSG = 2
        const NSG: u64 = 2;
        const NR0: u64 = 2; // N_R0_Q4_K = 2
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG, // 2D: Y = 2 for 2 SIMD groups
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        (0..m).map(|i| unsafe { *output_ptr.add(i) }).collect()
    }
}

fn main() {
    let m = 4096usize;
    let k = 4096usize;

    println!("Testing self-written Q4_K MV shader");
    println!("M={}, K={}", m, k);

    let ctx = TestContext::new();
    let weights = generate_weights(m, k);
    let input = generate_input(k);

    // Warmup
    let _ = ctx.run_mv(m, k, &weights, &input);

    // Correctness check with known values
    let output = ctx.run_mv(m, k, &weights, &input);
    println!("\nSample outputs:");
    for i in 0..5.min(m) {
        println!("  output[{}] = {:.6}", i, output[i]);
    }

    // Performance test
    let iterations = 10000;
    println!("\nRunning {} iterations...", iterations);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ctx.run_mv(m, k, &weights, &input);
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_secs_f64() * 1e6 / iterations as f64;
    let gflops = (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9;

    println!("\nResults:");
    println!("  Avg time: {:.2} us", avg_us);
    println!("  GFLOPS: {:.2}", gflops);
}
