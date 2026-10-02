// Back-to-back fair comparison in same process
use half::f16;
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i + j) % 64) as u8;
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

const FC_MUL_MV_NSG: NSUInteger = 600;
const FC_MUL_MV_NE12: NSUInteger = 602;
const FC_MUL_MV_R2: NSUInteger = 603;
const FC_MUL_MV_R3: NSUInteger = 604;

fn create_mv_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let nsg_ptr: *const c_void = &nsg as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            nsg_ptr,
            MTLDataType::Short,
            FC_MUL_MV_NSG,
        );
        let ne12_ptr: *const c_void = &ne12 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            ne12_ptr,
            MTLDataType::Short,
            FC_MUL_MV_NE12,
        );
        let r2_ptr: *const c_void = &r2 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            r2_ptr,
            MTLDataType::Short,
            FC_MUL_MV_R2,
        );
        let r3_ptr: *const c_void = &r3 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            r3_ptr,
            MTLDataType::Short,
            FC_MUL_MV_R3,
        );
    }
    cv
}

fn main() {
    let m = 4096usize;
    let k = 4096usize;
    let iterations = 10000;

    let weights = generate_q4_k_weights(m, k);
    let input: Vec<f32> = (0..k).map(|i| (i as f32).sin()).collect();

    println!("=== Back-to-Back Fair Comparison ===");
    println!("M={}, K={}, iterations={}\n", m, k, iterations);

    // Test 1: llama.cpp style (direct Metal API, no RefCell, no caching)
    {
        let device = Device::system_default().expect("No Metal device");
        let queue = device.new_command_queue();

        // Compile shader (same as llama.cpp)
        let source = include_str!("../../shaders/mul_mv_template.metal");
        let library = device
            .new_library_with_source(source, &CompileOptions::new())
            .expect("Failed to compile");
        let cv = create_mv_function_constants(2, 1, 1, 1);
        let kernel = library
            .get_function("kernel_mul_mv_q4_K_f32", Some(cv))
            .expect("Failed to get kernel");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .expect("Failed to create pipeline");

        // Create buffers ONCE (same as llama.cpp ggml_backend_tensor_set)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = device.new_buffer_with_data(
            weights.as_ptr() as *const c_void,
            weights_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device.new_buffer_with_data(
            input.as_ptr() as *const c_void,
            input_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = device.new_buffer(
            output_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Args (same as llama.cpp)
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;
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

        // Warmup
        {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            let nsg = 2u64;
            let nr0 = 2u64;
            let rows_per_tg = nsg * nr0;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (m as u64 + rows_per_tg - 1) / rows_per_tg,
                    height: 1,
                    depth: 1,
                },
                MTLSize { width: 32, height: nsg, depth: 1 },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Measure
        let start = Instant::now();
        for _ in 0..iterations {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            let nsg = 2u64;
            let nr0 = 2u64;
            let rows_per_tg = nsg * nr0;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (m as u64 + rows_per_tg - 1) / rows_per_tg,
                    height: 1,
                    depth: 1,
                },
                MTLSize { width: 32, height: nsg, depth: 1 },
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let elapsed = start.elapsed();
        let avg = elapsed.as_micros() as f64 / iterations as f64;
        let gflops = (m * k * 2) as f64 / (avg * 1e-6) / 1e9;
        println!("Raw Metal (llama.cpp style):");
        println!("  Avg: {:.2} us, GFLOPS: {:.2}", avg, gflops);
    }

    // Test 2: Rust MetalContext (with RefCell, Option, size checks)
    {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        // Warmup
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");

        let start = Instant::now();
        for _ in 0..iterations {
            let _ =
                ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");
        }
        let elapsed = start.elapsed();
        let avg = elapsed.as_micros() as f64 / iterations as f64;
        let gflops = (m * k * 2) as f64 / (avg * 1e-6) / 1e9;
        println!("\nRust MetalContext:");
        println!("  Avg: {:.2} us, GFLOPS: {:.2}", avg, gflops);
    }
}
