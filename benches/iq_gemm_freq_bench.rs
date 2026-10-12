// IQ GEMM benchmark with GPU frequency normalization
// Tests all IQ formats with alternating A-B-A-B pattern

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_stats};

const QK_K: usize = 256;
const QK4_NL: usize = 32;

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

// IQ2_S: d: u16, qs: [u8; 64], qh: [u8; 8], scales: [u8; 8] = 82 bytes
fn gen_iq2_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut qh = [0u8; 8];
            let mut scales = [0u8; 8];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ2S { d: f16::from_f32(1.0).to_bits(), qs, qh, scales }
        })
        .collect()
}

// IQ3_XXS: d: u16, qs: [u8; 96] = 98 bytes
fn gen_iq3_xxs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3XXS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 96];
            for j in 0..96 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ3XXS { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

// IQ3_S: d: u16, qs: [u8; 64], qh: [u8; 8], signs: [u8; 32], scales: [u8; 4] = 110 bytes
fn gen_iq3_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            let mut qh = [0u8; 8];
            let mut signs = [0u8; 32];
            let mut scales = [0u8; 4];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..32 {
                signs[j] = rng.r#gen::<u8>();
            }
            for j in 0..4 {
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ3S { d: f16::from_f32(1.0).to_bits(), qs, qh, signs, scales }
        })
        .collect()
}

// IQ4_NL: d: u16, qs: [u8; 16] = 18 bytes (32 elements per block!)
fn gen_iq4_nl(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4NL> {
    let nb = k / QK4_NL;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

// IQ4_XS: d: u16, scales_h: u16, scales_l: [u8; 4], qs: [u8; 128] = 136 bytes
fn gen_iq4_xs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4XS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 128];
            let mut scales_l = [0u8; 4];
            for j in 0..128 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..4 {
                scales_l[j] = rng.r#gen::<u8>();
            }
            BlockIQ4XS {
                d: f16::from_f32(1.0).to_bits(),
                scales_h: 0,
                scales_l,
                qs,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
#[repr(C)]
struct GemmArgs {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
}

const FC_MUL_MM: u64 = 700;

fn create_fc() -> metal::FunctionConstantValues {
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
    cv
}

fn bench_format(
    name: &str,
    kernel: &str,
    device: &Device,
    queue: &metal::CommandQueue,
    self_lib: &metal::Library,
    llama_lib: &metal::Library,
    weights: &[u8],
    input: &[u16],
    args: &GemmArgs,
    m: usize,
    n: usize,
    rounds: usize,
) {
    let cv = create_fc();

    let self_fn = self_lib.get_function(kernel, Some(cv.clone())).unwrap();
    let llama_fn = llama_lib.get_function(kernel, Some(cv.clone())).unwrap();

    let self_pipe =
        device.new_compute_pipeline_state_with_function(&self_fn).unwrap();
    let llama_pipe =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (input.len() * 2) as u64,
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

    // Warmup
    for _ in 0..5 {
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self_pipe);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgs>() as u64,
                args as *const _ as *const c_void,
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
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&llama_pipe);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgs>() as u64,
                args as *const _ as *const c_void,
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
    }

    // Verify correctness
    {
        let ptr_self = output_self.contents() as *const f32;
        let ptr_llama = output_llama.contents() as *const f32;
        let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
        let slice_llama =
            unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

        let max_diff = slice_self
            .iter()
            .zip(slice_llama.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        if max_diff > 1e-3 {
            println!("  {}: ✗ max_diff={:.6}", name, max_diff);
            return;
        }
    }

    // Alternating benchmark with GPU frequency normalization
    let mut self_total_ns = 0u64;
    let mut llama_total_ns = 0u64;
    let mut self_wins = 0;

    for _ in 0..rounds {
        let t0 = Instant::now();
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self_pipe);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgs>() as u64,
                args as *const _ as *const c_void,
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
        self_total_ns += t0.elapsed().as_nanos() as u64;

        let t1 = Instant::now();
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&llama_pipe);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgs>() as u64,
                args as *const _ as *const c_void,
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
        llama_total_ns += t1.elapsed().as_nanos() as u64;

        if self_total_ns < llama_total_ns {
            self_wins += 1;
        }
    }

    let gpu = get_gpu_stats();
    let freq_mhz = gpu.as_ref().map(|g| g.gpu_freq_mhz as f64).unwrap_or(1.0);

    let self_us = self_total_ns as f64 / rounds as f64 / 1000.0;
    let llama_us = llama_total_ns as f64 / rounds as f64 / 1000.0;

    let self_norm = self_us / freq_mhz;
    let llama_norm = llama_us / freq_mhz;

    let ratio_raw = self_us / llama_us * 100.0;
    let ratio_norm = self_norm / llama_norm * 100.0;

    println!(
        "  {}: {:.1}us vs {:.1}us ({:.1}%) | norm: {:.3} vs {:.3} ({:.1}%) | {}/{} wins",
        name,
        self_us,
        llama_us,
        ratio_raw,
        self_norm,
        llama_norm,
        ratio_norm,
        self_wins,
        rounds
    );
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let self_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_standalone.metallib")
        .unwrap();
    let llama_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_template.metallib")
        .unwrap();

    let m = 4096;
    let k = 4096;
    let n = 128;
    let seed = 42;
    let rounds = 100;

    println!("=== IQ GEMM Benchmark (GPU Freq Normalized) ===");
    println!("M={}, K={}, N={}, {} rounds", m, k, n, rounds);

    // GPU status
    let gpu = get_gpu_stats();
    if let Some(ref g) = gpu {
        println!("GPU Freq: {} MHz", g.gpu_freq_mhz);
    }
    println!();

    let input = gen_f16_input(k, n, seed);

    // IQ2_S
    let weights = gen_iq2_s(m, k, seed);
    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: 82,
        nb02: m as u64 * 82,
        nb03: 0,
        ne11: k as i32,
        nb10: 4,
        nb11: 4,
        nb12: 4 * k as u64,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
    };
    bench_format(
        "IQ2_S",
        "kernel_mul_mm_iq2_s_f32",
        &device,
        &queue,
        &self_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input,
        &args,
        m,
        n,
        rounds,
    );

    // IQ3_XXS
    let weights = gen_iq3_xxs(m, k, seed);
    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: 98,
        nb02: m as u64 * 98,
        nb03: 0,
        ne11: k as i32,
        nb10: 4,
        nb11: 4,
        nb12: 4 * k as u64,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
    };
    bench_format(
        "IQ3_XXS",
        "kernel_mul_mm_iq3_xxs_f32",
        &device,
        &queue,
        &self_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input,
        &args,
        m,
        n,
        rounds,
    );

    // IQ3_S
    let weights = gen_iq3_s(m, k, seed);
    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: 110,
        nb02: m as u64 * 110,
        nb03: 0,
        ne11: k as i32,
        nb10: 4,
        nb11: 4,
        nb12: 4 * k as u64,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
    };
    bench_format(
        "IQ3_S",
        "kernel_mul_mm_iq3_s_f32",
        &device,
        &queue,
        &self_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input,
        &args,
        m,
        n,
        rounds,
    );

    // IQ4_NL
    let weights = gen_iq4_nl(m, k, seed);
    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: 18,
        nb02: m as u64 * 18,
        nb03: 0,
        ne11: k as i32,
        nb10: 4,
        nb11: 4,
        nb12: 4 * k as u64,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
    };
    bench_format(
        "IQ4_NL",
        "kernel_mul_mm_iq4_nl_f32",
        &device,
        &queue,
        &self_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input,
        &args,
        m,
        n,
        rounds,
    );

    // IQ4_XS
    let weights = gen_iq4_xs(m, k, seed);
    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: 136,
        nb02: m as u64 * 136,
        nb03: 0,
        ne11: k as i32,
        nb10: 4,
        nb11: 4,
        nb12: 4 * k as u64,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
    };
    bench_format(
        "IQ4_XS",
        "kernel_mul_mm_iq4_xs_f32",
        &device,
        &queue,
        &self_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input,
        &args,
        m,
        n,
        rounds,
    );

    println!("\n=== Done ===");
}
