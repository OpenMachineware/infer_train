// Multi-size IQ GEMM benchmark
// Test all 9 IQ formats at multiple dimensions

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_stats};

const QK_K: usize = 256;

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
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

fn gen_iq1_m(m: usize, k: usize, seed: u64) -> Vec<BlockIQ1M> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            let mut qh = [0u8; 16];
            let mut scales = [0u8; 8];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..16 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ1M { qs, qh, scales }
        })
        .collect()
}

fn gen_iq2_xxs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2XXS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u16; 32];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u16>();
            }
            BlockIQ2XXS { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_iq2_xs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2XS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u16; 32];
            let mut scales = [0u8; 8];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u16>();
            }
            for j in 0..8 {
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ2XS { d: f16::from_f32(1.0).to_bits(), qs, scales }
        })
        .collect()
}

fn gen_iq2_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 64];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ2S { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_iq3_xxs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3XXS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u32; 8];
            for j in 0..8 {
                qs[j] = rng.r#gen::<u32>();
            }
            BlockIQ3XXS { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_iq3_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ3S { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_iq4_nl(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4NL> {
    let nb = k / 32; // IQ4_NL has 32 elements per block
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

fn gen_iq4_xs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4XS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u16; 68];
            for j in 0..68 {
                qs[j] = rng.r#gen::<u16>();
            }
            BlockIQ4XS { qs }
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

fn bench_format(
    name: &str,
    kernel_name: &str,
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    weights: &[u8],
    input_f16: &[u16],
    k: usize,
    m: usize,
    n: usize,
    block_size: usize,
    block_bytes: usize,
    rounds: usize,
) -> (f64, f64) {
    let queue = device.new_command_queue();

    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            FC_MUL_MM as u64,
        );
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            (FC_MUL_MM + 1) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 2) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 3) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 4) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 5) as u64,
        );
    }

    let my_fn = my_lib
        .get_function(kernel_name, Some(cv.clone()))
        .expect("Failed to get self kernel");
    let llama_fn = llama_lib
        .get_function(kernel_name, Some(cv))
        .expect("Failed to get llama kernel");

    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_fn)
        .expect("Failed to create self pipeline");
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_fn)
        .expect("Failed to create llama pipeline");

    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: (k / block_size * block_bytes) as u64,
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let args_buf = device.new_buffer_with_data(
        &args as *const GemmArgs as *const c_void,
        std::mem::size_of::<GemmArgs>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Alternating benchmark
    let mut my_times = Vec::with_capacity(rounds);
    let mut llama_times = Vec::with_capacity(rounds);

    for i in 0..rounds {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.compute_command_encoder();

        let (pipeline, output) = if i % 2 == 0 {
            (&my_pipeline, &output_buf)
        } else {
            (&llama_pipeline, &output_buf)
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&args_buf));
        encoder.set_buffer(1, Some(&weights_buf));
        encoder.set_buffer(2, Some(&input_buf));
        encoder.set_buffer(3, Some(output));
        encoder.set_threadgroup_memory_length(0, 8192);

        let threadgroup_size = MTLSize { width: 128, height: 1, depth: 1 };
        let threadgroups =
            MTLSize { width: (m + 63) / 64, height: (n + 31) / 32, depth: 1 };
        encoder.dispatch_threadgroups(threadgroups, threadgroup_size);
        encoder.end_encoding();

        let start = Instant::now();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        let elapsed = start.elapsed().as_micros() as f64;

        if i % 2 == 0 {
            my_times.push(elapsed);
        } else {
            llama_times.push(elapsed);
        }
    }

    let my_avg: f64 = my_times.iter().sum::<f64>() / my_times.len() as f64;
    let llama_avg: f64 =
        llama_times.iter().sum::<f64>() / llama_times.len() as f64;

    (my_avg, llama_avg)
}

fn main() {
    println!("=== IQ GEMM Multi-Size Benchmark ===\n");

    let device = Device::system_default().expect("No Metal device");

    if let Some(g) = get_gpu_stats() {
        print_stats("=== GPU Status ===", &g);
    }

    let my_lib = device
        .new_library_with_file(std::path::Path::new(
            "src/ops/metal/mul_mm_standalone.metallib",
        ))
        .expect("Failed to load self library");
    let llama_lib = device
        .new_library_with_file(std::path::Path::new(
            "src/ops/metal/mul_mm_template.metallib",
        ))
        .expect("Failed to load llama library");

    let rounds = 50usize;
    let seed = 42u64;

    // Test sizes: decode (N=1) with increasing M, K
    let sizes =
        [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)];

    println!("Testing {} rounds per format per size\n", rounds);

    let formats: [(&str, &str, usize, usize, fn(usize, usize, u64) -> Vec<u8>);
        9] = [
        ("IQ1_S", "kernel_mul_mm_iq1_s_f32", 256, 50, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq1_s(m, k, seed))
        }),
        ("IQ1_M", "kernel_mul_mm_iq1_m_f32", 256, 56, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq1_m(m, k, seed))
        }),
        ("IQ2_XXS", "kernel_mul_mm_iq2_xxs_f32", 256, 66, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq2_xxs(m, k, seed))
        }),
        ("IQ2_XS", "kernel_mul_mm_iq2_xs_f32", 256, 74, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq2_xs(m, k, seed))
        }),
        ("IQ2_S", "kernel_mul_mm_iq2_s_f32", 256, 82, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq2_s(m, k, seed))
        }),
        ("IQ3_XXS", "kernel_mul_mm_iq3_xxs_f32", 256, 98, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq3_xxs(m, k, seed))
        }),
        ("IQ3_S", "kernel_mul_mm_iq3_s_f32", 256, 110, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq3_s(m, k, seed))
        }),
        ("IQ4_NL", "kernel_mul_mm_iq4_nl_f32", 32, 18, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq4_nl(m, k, seed))
        }),
        ("IQ4_XS", "kernel_mul_mm_iq4_xs_f32", 256, 136, |m, k, seed| {
            bytemuck::cast_slice(&gen_iq4_xs(m, k, seed))
        }),
    ];

    println!(
        "| {:>8} | {:>10} | {:>10} | {:>8} |",
        "Format", "Self (µs)", "Llama (µs)", "Ratio"
    );
    println!("|----------|------------|------------|----------|");

    for (name, kernel, block_size, block_bytes, gen_fn) in formats.iter() {
        print!("| {:>8} |", name);
        for (m, k) in sizes.iter() {
            let input_f16 = gen_f16_input(*k, 1, seed);
            let weights = gen_fn(*m, *k, seed);

            let (self_us, llama_us) = bench_format(
                name,
                kernel,
                &device,
                &my_lib,
                &llama_lib,
                &weights,
                &input_f16,
                *k,
                *m,
                1,
                *block_size,
                *block_bytes,
                rounds,
            );

            let ratio = self_us / llama_us * 100.0;
            print!(" {:>8.1} / {:>8.1} ({:>5.1}%) |", self_us, llama_us, ratio);
        }
        println!();
    }

    println!("\n=== Done ===");
}
