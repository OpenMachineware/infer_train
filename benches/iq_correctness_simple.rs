// Simple correctness test for all 9 IQ kernels
// Minimal data (K=256, M=1, N=1), just verify kernels run without crash

use half::f16;
use metal::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

fn main() {
    println!("=== IQ Kernel Correctness Test ===");
    println!("Testing all 9 IQ formats with minimal data (K=256, M=1, N=1)\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Load shader with BOTH DECL and IMPL for Metal
    let gemm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let ggml_common = include_str!("../src/ops/metal/ggml-common.h");
    let ggml_common_metal = format!(
        "#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n{}",
        ggml_common
    );
    let combined = format!("{}\n{}", ggml_common_metal, gemm_source);

    println!("Compiling shader...");
    let compile_opts = CompileOptions::new();
    let lib = match device.new_library_with_source(&combined, &compile_opts) {
        Ok(l) => {
            println!("✓ Shader compiled successfully\n");
            l
        }
        Err(e) => {
            println!("✗ Compilation failed:");
            println!("{}", e);
            return;
        }
    };

    let mut rng = StdRng::seed_from_u64(42);

    // Test all 9 IQ formats
    let results = [
        test_format("IQ4_NL", test_iq4_nl, &device, &queue, &lib, &mut rng),
        test_format("IQ2_XXS", test_iq2_xxs, &device, &queue, &lib, &mut rng),
        test_format("IQ2_XS", test_iq2_xs, &device, &queue, &lib, &mut rng),
        test_format("IQ2_S", test_iq2_s, &device, &queue, &lib, &mut rng),
        test_format("IQ3_XXS", test_iq3_xxs, &device, &queue, &lib, &mut rng),
        test_format("IQ3_S", test_iq3_s, &device, &queue, &lib, &mut rng),
        test_format("IQ1_S", test_iq1_s, &device, &queue, &lib, &mut rng),
        test_format("IQ1_M", test_iq1_m, &device, &queue, &lib, &mut rng),
        test_format("IQ4_XS", test_iq4_xs, &device, &queue, &lib, &mut rng),
    ];

    println!("\n=== Summary ===");
    let passed = results.iter().filter(|&&r| r).count();
    println!("{}/{} kernels passed", passed, results.len());
}

type TestFn = fn(&Device, &CommandQueue, &Library, &mut StdRng) -> bool;

fn test_format(
    name: &str,
    test_fn: TestFn,
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    println!("=== Testing {} ===", name);
    let result = test_fn(device, queue, lib, rng);
    if result {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL\n");
    }
    result
}

fn make_args(k: i32, m: i32, n: i32, nb00: u64, nb01: u64) -> GemmArgs {
    GemmArgs {
        ne00: k,
        ne01: m,
        ne02: 1,
        nb00,
        nb01,
        nb02: 0,
        nb03: 0,
        ne10: k,
        ne11: n,
        ne12: 1,
        nb10: 4,
        nb11: (k as u64 * 4),
        nb12: 0,
        nb13: 0,
        ne0: m,
        ne1: n,
        nr0: 1,
        r2: 1,
        r3: 1,
    }
}

fn run_kernel(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    kernel_name: &str,
    args: &GemmArgs,
    weights: &[u8],
    input: &[f32],
    shmem_size: usize,
) -> Option<f32> {
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer =
        device.new_buffer(4u64, MTLResourceOptions::StorageModeShared);

    let cv = metal::FunctionConstantValues::new();
    let nsg: i16 = 2;
    let ne12: i16 = 1;
    let r2: i16 = 1;
    let r3: i16 = 1;
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const _ as *const c_void,
            metal::MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const _ as *const c_void,
            metal::MTLDataType::Short,
            604,
        );
    }

    let func = lib.get_function(kernel_name, Some(cv)).ok()?;
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).ok()?;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<GemmArgs>() as u64,
        args as *const _ as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
    enc.set_buffer(3, Some(&output_buffer), 0);
    if shmem_size > 0 {
        enc.set_threadgroup_memory_length(0, shmem_size as u64);
    }
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(32, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = unsafe { *(output_buffer.contents() as *const f32) };
    Some(result)
}

#[repr(C)]
struct GemmArgs {
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

// IQ4_NL: QK=32, block = d(2) + qs[16] = 18 bytes
fn test_iq4_nl(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let num_blocks = k / 32;
    let mut weights = Vec::new();
    for _ in 0..num_blocks {
        weights.push(0x00);
        weights.push(0x3C); // d = 1.0
        for _ in 0..16 {
            weights.push(rng.r#gen::<u8>());
        }
    }
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 18, 18 * num_blocks as u64);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq4_nl_f32",
        &args,
        &weights,
        &input,
        0,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}

// IQ2_XXS: block = d(2) + qs[32](64) = 66 bytes
fn test_iq2_xxs(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let mut weights = vec![0x00, 0x3C]; // d = 1.0
    for _ in 0..32 {
        let v = rng.r#gen::<u16>();
        weights.push((v & 0xFF) as u8);
        weights.push((v >> 8) as u8);
    }
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 66, 66);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq2_xxs_f32",
        &args,
        &weights,
        &input,
        2208,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}

// IQ2_XS: block = d(2) + qs[32](64) + scales[8] = 74 bytes
fn test_iq2_xs(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let mut weights = vec![0x00, 0x3C];
    for _ in 0..32 {
        let v = rng.r#gen::<u16>();
        weights.push((v & 0xFF) as u8);
        weights.push((v >> 8) as u8);
    }
    for _ in 0..8 {
        weights.push(rng.r#gen::<u8>());
    }
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 74, 74);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq2_xs_f32",
        &args,
        &weights,
        &input,
        2208,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}

// IQ2_S: block = d(2) + qs[64] + qh[8] + scales[8] = 82 bytes
fn test_iq2_s(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let mut weights = vec![0x00, 0x3C];
    for _ in 0..64 {
        weights.push(rng.r#gen::<u8>());
    }
    for _ in 0..8 {
        weights.push(rng.r#gen::<u8>());
    }
    for _ in 0..8 {
        weights.push(rng.r#gen::<u8>());
    }
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 82, 82);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq2_s_f32",
        &args,
        &weights,
        &input,
        2208,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}

// IQ3_XXS: block = d(2) + qs[96] = 98 bytes
fn test_iq3_xxs(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let mut weights = vec![0x00, 0x3C];
    for _ in 0..96 {
        weights.push(rng.r#gen::<u8>());
    }
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 98, 98);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq3_xxs_f32",
        &args,
        &weights,
        &input,
        2208,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}

// IQ3_S: block = d(2) + qs[64] + qh[8] + signs[32] + scales[4] = 110 bytes
fn test_iq3_s(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let mut weights = vec![0x00, 0x3C];
    for _ in 0..64 {
        weights.push(rng.r#gen::<u8>());
    }
    for _ in 0..8 {
        weights.push(rng.r#gen::<u8>());
    }
    for _ in 0..32 {
        weights.push(rng.r#gen::<u8>());
    }
    for _ in 0..4 {
        weights.push(rng.r#gen::<u8>());
    }
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 110, 110);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq3_s_f32",
        &args,
        &weights,
        &input,
        2208,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}

// IQ1_S: block = d(2) + qs[32] + qh[16] = 50 bytes
fn test_iq1_s(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let mut weights = vec![0x00, 0x3C];
    for _ in 0..32 {
        weights.push(rng.r#gen::<u8>());
    }
    for _ in 0..16 {
        let v = rng.r#gen::<u16>();
        weights.push((v & 0xFF) as u8);
        weights.push((v >> 8) as u8);
    }
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 50, 50);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq1_s_f32",
        &args,
        &weights,
        &input,
        2208,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}

// IQ1_M: block = qs[32] + qh[16] + scales[8] = 56 bytes (no d!)
fn test_iq1_m(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let mut weights = Vec::new();
    for _ in 0..32 {
        weights.push(rng.r#gen::<u8>());
    }
    for _ in 0..16 {
        let v = rng.r#gen::<u16>();
        weights.push((v & 0xFF) as u8);
        weights.push((v >> 8) as u8);
    }
    for _ in 0..8 {
        weights.push(rng.r#gen::<u8>());
    }
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 56, 56);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq1_m_f32",
        &args,
        &weights,
        &input,
        2208,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}

// IQ4_XS: block = d(2) + scales_h(2) + scales_l[4] + qs[128] = 136 bytes
fn test_iq4_xs(
    device: &Device,
    queue: &CommandQueue,
    lib: &Library,
    rng: &mut StdRng,
) -> bool {
    let k = 256;
    let mut weights = vec![0x00, 0x3C]; // d = 1.0
    weights.push(0x00);
    weights.push(0x00); // scales_h
    for _ in 0..4 {
        weights.push(rng.r#gen::<u8>());
    } // scales_l
    for _ in 0..128 {
        weights.push(rng.r#gen::<u8>());
    } // qs
    let input: Vec<f32> = (0..k).map(|i| 0.01 * (i as f32 % 10.0)).collect();
    let args = make_args(k, 1, 1, 136, 136);
    let result = run_kernel(
        device,
        queue,
        lib,
        "kernel_mul_mm_iq4_xs_f32",
        &args,
        &weights,
        &input,
        64,
    );
    result
        .map(|r| {
            println!("Result: {:.6}", r);
            r.is_finite()
        })
        .unwrap_or(false)
}
