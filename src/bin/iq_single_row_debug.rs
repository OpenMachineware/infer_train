// Debug IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS correctness
// Single row test with direct output comparison

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

const QK_K: usize = 256;

fn gen_f32_input(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

fn gen_iq1_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ1S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 8];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u16>();
            }
            BlockIQ1S { d: f16::from_f32(1.0).to_bits(), qs, qh }
        })
        .collect()
}

#[repr(C)]
struct GemmArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

const FC_MUL_MM: u64 = 700;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let self_lib = device
        .new_library_with_file("shaders/mul_mm_standalone.metallib")
        .unwrap();
    let llama_lib = device
        .new_library_with_file("shaders/mul_mm_template.metallib")
        .unwrap();

    let m = 1; // Single row
    let k = 256; // Single block
    let n = 32;
    let seed = 42;

    println!("=== IQ Single Row Debug ===");
    println!("M={}, K={}, N={}\n", m, k, n);

    let input = gen_f32_input(k, n, seed);

    // IQ1_S test
    let weights = gen_iq1_s(m, k, seed);
    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: 50,
        nb02: m as u64 * 50,
        nb03: 0,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    let cv = metal::FunctionConstantValues::new();
    unsafe {
        let bc = false;
        let one: i16 = 1;
        cv.set_constant_value_at_index(
            &bc as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            FC_MUL_MM,
        );
        cv.set_constant_value_at_index(
            &bc as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            FC_MUL_MM + 1,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            metal::MTLDataType::Short,
            FC_MUL_MM + 2,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            metal::MTLDataType::Short,
            FC_MUL_MM + 3,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            metal::MTLDataType::Short,
            FC_MUL_MM + 4,
        );
        cv.set_constant_value_at_index(
            &one as *const _ as *const c_void,
            metal::MTLDataType::Short,
            FC_MUL_MM + 5,
        );
    }

    let self_fn = self_lib
        .get_function("kernel_mul_mm_iq1_s_f32", Some(cv.clone()))
        .unwrap();
    let llama_fn = llama_lib
        .get_function("kernel_mul_mm_iq1_s_f32", Some(cv.clone()))
        .unwrap();

    let self_pipe =
        device.new_compute_pipeline_state_with_function(&self_fn).unwrap();
    let llama_pipe =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64 * 50,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (input.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_self = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let tg_size = MTLSize::new(128, 1, 1);
    let tg_count =
        MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);

    // Run self kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self_pipe);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_self), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run llama kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipe);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare outputs
    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    println!("IQ1_S output comparison (first 16 values):");
    println!("Index | Self      | Llama     | Diff");
    println!("------|-----------|-----------|--------");
    for i in 0..16.min(m * n) {
        let diff = (slice_self[i] - slice_llama[i]).abs();
        println!(
            "{:5} | {:9.4} | {:9.4} | {:7.4}",
            i, slice_self[i], slice_llama[i], diff
        );
    }

    let max_diff = slice_self
        .iter()
        .zip(slice_llama.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\nmax_diff = {:.6}", max_diff);
}
