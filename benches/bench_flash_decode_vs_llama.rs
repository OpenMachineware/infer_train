// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Alternating benchmark: Our Adaptive vs llama.cpp Flash Attention Decode

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 200;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load our kernels
    let our_source = include_str!("../src/ops/metal/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    // Load llama.cpp kernels
    let llama_source =
        load_llama_kernels().expect("Failed to load llama.cpp kernels");
    let llama_library = device
        .new_library_with_source(&llama_source, &options)
        .expect("Failed to compile llama.cpp kernels");

    println!("=== Flash Attention Decode: Ours vs llama.cpp ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    // Test formats and seq_k sizes
    let formats = ["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"];
    let seq_ks = [256, 512, 1024, 2048];

    for format in &formats {
        println!("\n=== Format: {} ===", format);
        println!(
            "{:>8} {:>6} {:>12} {:>12} {:>10}",
            "seq_k", "NSG", "Ours (µs)", "llama (µs)", "Ratio"
        );
        println!("{}", "-".repeat(55));

        for seq_k in &seq_ks {
            let result = benchmark_format_seq_k(
                &device,
                &queue,
                &our_library,
                &llama_library,
                format,
                *seq_k,
            );

            println!(
                "{:>8} {:>6} {:>12.2} {:>12.2} {:>9.1}%",
                seq_k,
                result.nsg,
                result.our_time_us,
                result.llama_time_us,
                result.ratio_pct
            );
        }
    }
}

struct BenchmarkResult {
    nsg: usize,
    our_time_us: f64,
    llama_time_us: f64,
    ratio_pct: f64,
}

fn benchmark_format_seq_k(
    device: &Device,
    queue: &CommandQueue,
    our_library: &Library,
    llama_library: &Library,
    format: &str,
    seq_k: usize,
) -> BenchmarkResult {
    // Create F16 data for both
    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; seq_k * NUM_HEADS * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3400; seq_k * NUM_HEADS * HEAD_DIM];

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_data.as_ptr() as *const _,
        (k_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_data.as_ptr() as *const _,
        (v_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // NSG based on seq_k
    let nsg = if seq_k <= 128 {
        1
    } else if seq_k <= 256 {
        2
    } else if seq_k <= 512 {
        4
    } else {
        8
    };
    let shmem_size = nsg * (2 + HEAD_DIM) * 4;

    // Get kernels
    let our_kernel_name = format!("flash_decode_attention_{}_adaptive", format);
    let our_kernel = our_library.get_function(&our_kernel_name, None).unwrap();
    let our_pipeline =
        device.new_compute_pipeline_state_with_function(&our_kernel).unwrap();

    // llama.cpp kernel name depends on format
    let llama_kernel_name = get_llama_kernel_name(format);
    let llama_kernel =
        llama_library.get_function(&llama_kernel_name, None).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_kernel).unwrap();

    // Args struct for our kernel
    #[repr(C)]
    struct OurArgs {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        scale: f32,
        q_stride_b: u32,
        q_stride_h: u32,
        q_stride_s: u32,
        k_stride_b: u32,
        k_stride_h: u32,
        k_stride_s: u32,
        v_stride_b: u32,
        v_stride_h: u32,
        v_stride_s: u32,
        o_stride_b: u32,
        o_stride_h: u32,
        o_stride_s: u32,
    }

    let our_args = OurArgs {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        q_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
        q_stride_h: HEAD_DIM as u32,
        q_stride_s: HEAD_DIM as u32,
        k_stride_b: (NUM_HEADS * seq_k * HEAD_DIM * 2) as u32,
        k_stride_h: (seq_k * HEAD_DIM * 2) as u32,
        k_stride_s: (HEAD_DIM * 2) as u32,
        v_stride_b: (NUM_HEADS * seq_k * HEAD_DIM * 2) as u32,
        v_stride_h: (seq_k * HEAD_DIM * 2) as u32,
        v_stride_s: (HEAD_DIM * 2) as u32,
        o_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
        o_stride_h: HEAD_DIM as u32,
        o_stride_s: HEAD_DIM as u32,
    };

    let our_args_bytes = unsafe {
        std::slice::from_raw_parts(
            &our_args as *const _ as *const u8,
            std::mem::size_of::<OurArgs>(),
        )
    };

    // llama.cpp args (ggml_metal_kargs_flash_attn_ext_vec)
    // This is a simplified version - llama.cpp has many more fields
    #[repr(C)]
    struct LlamaArgs {
        ne01: u32,
        ne02: u32,
        ne03: u32,
        nb01: u32,
        nb02: u32,
        nb03: u32,
        ne11: u32,
        ne_12_2: u32,
        ne_12_3: u32,
        ns10: u32,
        nb11: u32,
        nb12: u32,
        nb13: u32,
        ns20: u32,
        nb21: u32,
        nb22: u32,
        nb23: u32,
        ne31: u32,
        ne32: u32,
        ne33: u32,
        nb31: u32,
        nb32: u32,
        nb33: u32,
        ne1: u32,
        ne2: u32,
        ne3: u32,
        scale: f32,
        max_bias: f32,
        m0: u32,
        m1: u32,
        n_head_log2: u32,
        logit_softcap: f32,
        n_kv_max_padded: u32,
    }

    let llama_args = LlamaArgs {
        ne01: 1,
        ne02: NUM_HEADS as u32,
        ne03: 1,
        nb01: (HEAD_DIM * 4) as u32,
        nb02: (NUM_HEADS * HEAD_DIM * 4) as u32,
        nb03: (NUM_HEADS * HEAD_DIM * 4) as u32,
        ne11: seq_k as u32,
        ne_12_2: 1,
        ne_12_3: 1,
        ns10: 1,
        nb11: (HEAD_DIM * 2) as u32,
        nb12: (seq_k * HEAD_DIM * 2) as u32,
        nb13: (NUM_HEADS * seq_k * HEAD_DIM * 2) as u32,
        ns20: 1,
        nb21: (HEAD_DIM * 2) as u32,
        nb22: (seq_k * HEAD_DIM * 2) as u32,
        nb23: (NUM_HEADS * seq_k * HEAD_DIM * 2) as u32,
        ne31: 1,
        ne32: NUM_HEADS as u32,
        ne33: 1,
        nb31: (HEAD_DIM * 4) as u32,
        nb32: (NUM_HEADS * HEAD_DIM * 4) as u32,
        nb33: (NUM_HEADS * HEAD_DIM * 4) as u32,
        ne1: 1,
        ne2: NUM_HEADS as u32,
        ne3: 1,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        max_bias: 0.0,
        m0: 0,
        m1: 0,
        n_head_log2: 0,
        logit_softcap: 0.0,
        n_kv_max_padded: ((seq_k + 31) / 32 * 32) as u32,
    };

    let llama_args_bytes = unsafe {
        std::slice::from_raw_parts(
            &llama_args as *const _ as *const u8,
            std::mem::size_of::<LlamaArgs>(),
        )
    };

    // Warmup
    for _ in 0..5 {
        run_our_kernel(
            &queue,
            &our_pipeline,
            our_args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            nsg,
            shmem_size,
        );
        run_llama_kernel(
            &queue,
            &llama_pipeline,
            llama_args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
        );
    }

    // Alternating benchmark
    let mut our_times = Vec::with_capacity(NUM_ROUNDS);
    let mut llama_times = Vec::with_capacity(NUM_ROUNDS);

    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        run_our_kernel(
            &queue,
            &our_pipeline,
            our_args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            nsg,
            shmem_size,
        );
        our_times.push(t0.elapsed().as_nanos() as f64);

        let t1 = Instant::now();
        run_llama_kernel(
            &queue,
            &llama_pipeline,
            llama_args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
        );
        llama_times.push(t1.elapsed().as_nanos() as f64);
    }

    our_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let our_median = our_times[NUM_ROUNDS / 2] / 1000.0; // µs
    let llama_median = llama_times[NUM_ROUNDS / 2] / 1000.0; // µs
    let ratio = our_median / llama_median * 100.0;

    BenchmarkResult {
        nsg,
        our_time_us: our_median,
        llama_time_us: llama_median,
        ratio_pct: ratio,
    }
}

fn run_our_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args_bytes: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    num_heads: usize,
    nsg: usize,
    shmem_size: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args_bytes.len() as u64, args_bytes.as_ptr() as *const _);
    enc.set_buffer(1, Some(q_buf), 0);
    enc.set_buffer(2, Some(k_buf), 0);
    enc.set_buffer(3, Some(v_buf), 0);
    enc.set_buffer(4, Some(out_buf), 0);
    if shmem_size > 0 {
        enc.set_threadgroup_memory_length(0, shmem_size as u64);
    }
    enc.dispatch_thread_groups(
        MTLSize::new(1, num_heads as u64, 1),
        MTLSize::new(32, nsg as u64, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn run_llama_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args_bytes: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    num_heads: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args_bytes.len() as u64, args_bytes.as_ptr() as *const _);
    enc.set_buffer(1, Some(q_buf), 0);
    enc.set_buffer(2, Some(k_buf), 0);
    enc.set_buffer(3, Some(v_buf), 0);
    enc.set_buffer(4, Some(out_buf), 0);
    // llama.cpp uses 128 threads (4 SIMD groups)
    enc.dispatch_thread_groups(
        MTLSize::new(1, num_heads as u64, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn get_llama_kernel_name(format: &str) -> String {
    match format {
        "f16" => "kernel_flash_attn_ext_vec_f16_dk128_dv128".to_string(),
        "q4_0" => "kernel_flash_attn_ext_vec_q4_0_dk128_dv128".to_string(),
        "q4_1" => "kernel_flash_attn_ext_vec_q4_1_dk128_dv128".to_string(),
        "q5_0" => "kernel_flash_attn_ext_vec_q5_0_dk128_dv128".to_string(),
        "q5_1" => "kernel_flash_attn_ext_vec_q5_1_dk128_dv128".to_string(),
        "q8_0" => "kernel_flash_attn_ext_vec_q8_0_dk128_dv128".to_string(),
        _ => panic!("Unknown format: {}", format),
    }
}

fn load_llama_kernels() -> Result<String, std::io::Error> {
    let base = "llama.cpp-0.6.0/ggml/src/ggml-metal/kernels";
    let impl_base = "llama.cpp-0.6.0/ggml/src/ggml-metal";
    let common_base = "llama.cpp-0.6.0/ggml/src";

    // Load impl header (needed by common.h)
    let impl_header =
        std::fs::read_to_string(&format!("{}/ggml-metal-impl.h", impl_base))?;

    // Load ggml-common.h (needed by dequantize.h)
    let ggml_common =
        std::fs::read_to_string(&format!("{}/ggml-common.h", common_base))?;

    // Load required files and strip includes/pragma
    let common = std::fs::read_to_string(&format!("{}/common.h", base))?;
    let dequantize =
        std::fs::read_to_string(&format!("{}/dequantize.h", base))?;
    let fa_common =
        std::fs::read_to_string(&format!("{}/fa_common.metal", base))?;
    let fa_aux = std::fs::read_to_string(&format!("{}/fa_aux.metal", base))?;
    let fa_vec_common =
        std::fs::read_to_string(&format!("{}/fa_vec_common.metal", base))?;
    let fa_vec_f16 =
        std::fs::read_to_string(&format!("{}/fa_vec_f16.metal", base))?;
    let fa_vec_q4_0 =
        std::fs::read_to_string(&format!("{}/fa_vec_q4_0.metal", base))?;
    let fa_vec_q4_1 =
        std::fs::read_to_string(&format!("{}/fa_vec_q4_1.metal", base))?;
    let fa_vec_q5_0 =
        std::fs::read_to_string(&format!("{}/fa_vec_q5_0.metal", base))?;
    let fa_vec_q5_1 =
        std::fs::read_to_string(&format!("{}/fa_vec_q5_1.metal", base))?;
    let fa_vec_q8_0 =
        std::fs::read_to_string(&format!("{}/fa_vec_q8_0.metal", base))?;

    // Helper to strip includes and pragma once
    fn strip_includes(content: &str) -> String {
        content
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.starts_with("#include")
                    && !trimmed.starts_with("#pragma once")
                    && !trimmed.starts_with("__embed_")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    Ok(format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        impl_header,
        ggml_common,
        strip_includes(&common),
        strip_includes(&dequantize),
        fa_common,
        fa_aux,
        fa_vec_common,
        fa_vec_f16,
        fa_vec_q4_0,
        fa_vec_q4_1,
        fa_vec_q5_0,
        fa_vec_q5_1,
        fa_vec_q8_0
    ))
}
