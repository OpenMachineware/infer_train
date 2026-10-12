// Correctness test for IQ GEMM kernels
// Compare self-written mul_mm_standalone.metal vs llama.cpp mul_mm_template.metal
// Uses quantization functions to generate proper test data

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

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
            let mut qh = [0u8; 8];
            let mut scales = [0u8; 8];
            for j in 0..64 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u8>();
                scales[j] = rng.r#gen::<u8>();
            }
            BlockIQ2S { d: f16::from_f32(1.0).to_bits(), qs, qh, scales }
        })
        .collect()
}

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

fn gen_iq4_nl(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
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
                scales_h: f16::from_f32(1.0).to_bits(),
                scales_l,
                qs,
            }
        })
        .collect()
}

// Parameter structure for mul_mm kernels (matches ggml_metal_kargs_mul_mm)
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

fn test_format(
    name: &str,
    kernel_name: &str,
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    weights: &[u8],
    input_f16: &[u16],
    args: &GemmArgs,
    m: usize,
    n: usize,
    block_size: usize,
) -> bool {
    let queue = device.new_command_queue();

    // Function constants (FC_MUL_MM = 700 for both)
    const FC_MUL_MM: usize = 700;
    let cv = metal::FunctionConstantValues::new();
    let one_val: i16 = 1;
    let false_val: bool = false;
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

    // Get both functions (same kernel name)
    let my_fn = my_lib.get_function(kernel_name, Some(cv.clone())).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();

    let llama_fn = llama_lib.get_function(kernel_name, Some(cv)).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    // Create buffers
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_my = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Threadgroup dispatch
    // Kernel expects: tgpig.x = row dimension (r0), tgpig.y = column dimension (r1)
    let rows_per_tg = 64; // NSG * nr0 = 4 * 16
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((m + rows_per_tg - 1) / rows_per_tg) as u64, // X dimension = rows
        n as u64,                                     // Y dimension = columns
        1,
    );

    // Run self-written kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_my), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run llama.cpp kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare outputs
    let ptr_my = output_my.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_my = unsafe { std::slice::from_raw_parts(ptr_my, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    let non_zero_my: usize = slice_my.iter().filter(|&&x| x != 0.0).count();
    let non_zero_llama: usize =
        slice_llama.iter().filter(|&&x| x != 0.0).count();

    let mut max_diff = 0.0f32;
    let mut diff_count = 0usize;
    let tolerance = 1e-3f32;

    for (a, b) in slice_my.iter().zip(slice_llama.iter()) {
        let diff = (a - b).abs();
        if diff > tolerance {
            diff_count += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    let sum_my: f64 = slice_my.iter().map(|&x| x as f64).sum();
    let sum_llama: f64 = slice_llama.iter().map(|&x| x as f64).sum();

    let passed = diff_count == 0 && non_zero_my > 0 && non_zero_llama > 0;

    println!("  {}", name);
    println!("    non_zero: self={}, llama={}", non_zero_my, non_zero_llama);
    println!(
        "    max_diff: {:.6}, diff_count: {} (> {})",
        max_diff, diff_count, tolerance
    );
    println!("    sum: self={:.2}, llama={:.2}", sum_my, sum_llama);

    if passed {
        println!("    ✅ PASSED");
    } else {
        println!("    ❌ FAILED");
    }

    passed
}

fn main() {
    println!("=== IQ GEMM Correctness Test ===");
    println!(
        "Comparing self-written mul_mm_standalone.metal vs llama.cpp mul_mm_template.metal\n"
    );

    let device = Device::system_default().expect("No Metal device");

    // Test dimensions: start small to verify kernel runs
    let m = 64usize;
    let k = 256usize; // Single K-quant block
    let n = 16usize; // Small N for GEMM
    let seed = 42u64;

    println!("M={}, K={}, N={} (conservative test)\n", m, k, n);

    // Load self-written mul_mm_standalone.metal
    // Need to define GGML_COMMON_IMPL_METAL before including ggml-common.h for correct grid tables
    let ggml_common = include_str!("../src/ops/metal/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_IMPL_METAL\n{}", ggml_common);
    let standalone_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");

    // Remove the #include line from standalone_source since we're prepending the content
    // Load pre-compiled metallib files
    let my_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_standalone.metallib");
    let my_lib = device
        .new_library_with_file(my_lib_path)
        .expect("Failed to load pre-compiled standalone GEMM library");

    let llama_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load pre-compiled llama.cpp template library");

    // Generate input
    let input_f16 = gen_f16_input(k, n, seed);

    let mut all_passed = true;

    // Helper to create args
    let make_args = |k: usize,
                     m: usize,
                     n: usize,
                     block_size: usize,
                     block_bytes: usize|
     -> GemmArgs {
        GemmArgs {
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
        }
    };

    // Test all 9 formats
    println!("=== IQ Format GEMM Correctness ===\n");

    // IQ1_S (256-element block, 50 bytes)
    let weights = gen_iq1_s(m, k, seed);
    let args = make_args(k, m, n, 256, 50);
    all_passed &= test_format(
        "IQ1_S",
        "kernel_mul_mm_iq1_s_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        256,
    );

    // IQ1_M (256-element block, 56 bytes)
    let weights = gen_iq1_m(m, k, seed);
    let args = make_args(k, m, n, 256, 56);
    all_passed &= test_format(
        "IQ1_M",
        "kernel_mul_mm_iq1_m_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        256,
    );

    // IQ2_XXS (256-element block, 66 bytes)
    let weights = gen_iq2_xxs(m, k, seed);
    let args = make_args(k, m, n, 256, 66);
    all_passed &= test_format(
        "IQ2_XXS",
        "kernel_mul_mm_iq2_xxs_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        256,
    );

    // IQ2_XS (256-element block, 74 bytes)
    let weights = gen_iq2_xs(m, k, seed);
    let args = make_args(k, m, n, 256, 74);
    all_passed &= test_format(
        "IQ2_XS",
        "kernel_mul_mm_iq2_xs_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        256,
    );

    // IQ2_S (256-element block, 82 bytes)
    let weights = gen_iq2_s(m, k, seed);
    let args = make_args(k, m, n, 256, 82);
    all_passed &= test_format(
        "IQ2_S",
        "kernel_mul_mm_iq2_s_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        256,
    );

    // IQ3_XXS (256-element block, 98 bytes)
    let weights = gen_iq3_xxs(m, k, seed);
    let args = make_args(k, m, n, 256, 98);
    all_passed &= test_format(
        "IQ3_XXS",
        "kernel_mul_mm_iq3_xxs_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        256,
    );

    // IQ3_S (256-element block, 110 bytes)
    let weights = gen_iq3_s(m, k, seed);
    let args = make_args(k, m, n, 256, 110);
    all_passed &= test_format(
        "IQ3_S",
        "kernel_mul_mm_iq3_s_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        256,
    );

    // IQ4_NL (32-element block, 18 bytes)
    let weights = gen_iq4_nl(m, k, seed);
    let args = make_args(k, m, n, 32, 18);
    all_passed &= test_format(
        "IQ4_NL",
        "kernel_mul_mm_iq4_nl_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        32,
    );

    // IQ4_XS (256-element block, 136 bytes)
    let weights = gen_iq4_xs(m, k, seed);
    let args = make_args(k, m, n, 256, 136);
    all_passed &= test_format(
        "IQ4_XS",
        "kernel_mul_mm_iq4_xs_f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        256,
    );

    println!();
    if all_passed {
        println!("=== All tests PASSED ✅ ===");
    } else {
        println!("=== Some tests FAILED ❌ ===");
    }
}
