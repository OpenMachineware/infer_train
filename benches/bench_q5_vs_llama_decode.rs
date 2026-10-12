// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Direct comparison: Our Multi-WG Q5 Flash Decode vs llama.cpp decode kernel
// Uses the correct llama.cpp decode kernel (kernel_flash_attn_ext_q5_*, not fa_vec)

use half::f16;
use infer_train::quant::kv_cache::{
    quantize_row_q5_0, quantize_row_q5_1, BlockQ5_0, BlockQ5_1,
};
use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 200;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load our kernels
    let our_source =
        include_str!("../src/ops/metal/flash_decode_multi_wg.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library = match device.new_library_with_source(our_source, &options)
    {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("Our kernel compilation error: {}", e);
            std::process::exit(1);
        }
    };

    // Load llama.cpp kernels
    let llama_source = match load_llama_decode_kernels() {
        Ok(src) => src,
        Err(e) => {
            eprintln!("Failed to load llama.cpp kernels: {}", e);
            std::process::exit(1);
        }
    };
    let llama_library =
        match device.new_library_with_source(&llama_source, &options) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("llama.cpp kernel compilation error: {}", e);
                std::process::exit(1);
            }
        };

    println!("=== Q5 Flash Decode: Ours vs llama.cpp Decode Kernel ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );
    println!("Using llama.cpp decode kernel (kernel_flash_attn_ext_q5_*), NOT prefill (fa_vec)\n");

    let formats = ["q5_0", "q5_1"];
    let seq_ks = [256, 512, 1024, 2048];

    println!("| Format | seq_k | Ours (G) | llama (G) | Ratio | Status |");
    println!("|--------|-------|----------|-----------|-------|--------|");

    for format in &formats {
        for &seq_k in &seq_ks {
            let our_gflops =
                bench_our_kernel(&device, &queue, &our_library, format, seq_k);
            let llama_gflops = bench_llama_kernel(
                &device,
                &queue,
                &llama_library,
                format,
                seq_k,
            );
            let ratio = our_gflops / llama_gflops * 100.0;
            let status = if ratio >= 100.0 { "✅" } else { "❌" };
            println!(
                "| {:6} | {:5} | {:8.2} | {:9.2} | {:5.1}% | {}      |",
                format, seq_k, our_gflops, llama_gflops, ratio, status
            );
        }
        println!("|--------|-------|----------|-----------|-------|--------|");
    }
}

fn load_llama_decode_kernels() -> Result<String, std::io::Error> {
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

    let common = std::fs::read_to_string(&format!("{}/common.h", base))?;
    let dequantize =
        std::fs::read_to_string(&format!("{}/dequantize.h", base))?;
    let fa_common =
        std::fs::read_to_string(&format!("{}/fa_common.metal", base))?;
    let fa_q5_0 = std::fs::read_to_string(&format!("{}/fa_q5_0.metal", base))?;
    let fa_q5_1 = std::fs::read_to_string(&format!("{}/fa_q5_1.metal", base))?;

    Ok(format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        impl_header,
        ggml_common,
        strip_includes(&common),
        strip_includes(&dequantize),
        fa_common,
        fa_q5_0,
        fa_q5_1
    ))
}

fn bench_our_kernel(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
) -> f64 {
    let nwgs = [8, 16, 32];
    let mut best_gflops = 0.0;

    for &nwg in &nwgs {
        let kernel_name =
            format!("flash_decode_multi_wg_partial_{}_c64_nwg{}", format, nwg);
        if let Ok(func) = library.get_function(&kernel_name, None) {
            let gflops = bench_our_with_nwg(
                device, queue, library, format, seq_k, nwg, &func,
            );
            if gflops > best_gflops {
                best_gflops = gflops;
            }
        }
    }

    best_gflops
}

fn bench_our_with_nwg(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
    nwg: usize,
    func: &Function,
) -> f64 {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;
    let src_data: Vec<f32> = vec![0.5; num_elements];

    let (k_buf, v_buf, _size) = match format {
        "q5_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_0(&src_data, &mut blocks);
            let size = num_blocks * 22;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, size)
        }
        "q5_1" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_1> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_1(&src_data, &mut blocks);
            let size = num_blocks * 24;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, size)
        }
        _ => panic!("Unknown format"),
    };

    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_buf = device.new_buffer(
        (NUM_HEADS * 16 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    let scale_buf = device.new_buffer_with_data(
        &scale as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_k_buf = device.new_buffer_with_data(
        &(seq_k as u32) as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let pipeline =
        device.new_compute_pipeline_state_with_function(func).unwrap();

    let finalize_func = library
        .get_function("flash_decode_multi_wg_finalize_f16_nwg8", None)
        .unwrap();
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(&finalize_func)
        .unwrap();

    let start = Instant::now();
    for _ in 0..NUM_ROUNDS {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&seq_k_buf), 0);
        encoder.set_buffer(1, Some(&scale_buf), 0);
        encoder.set_buffer(2, Some(&q_buf), 0);
        encoder.set_buffer(3, Some(&k_buf), 0);
        encoder.set_buffer(4, Some(&v_buf), 0);
        encoder.set_buffer(5, Some(&out_buf), 0);
        encoder.set_buffer(6, Some(&partial_buf), 0);

        let threads_per_group = MTLSize::new(32, 1, 1);
        let num_groups = MTLSize::new(nwg as u64, NUM_HEADS as u64, 1);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        let encoder2 = cmd_buf.new_compute_command_encoder();
        encoder2.set_compute_pipeline_state(&finalize_pipeline);
        encoder2.set_buffer(0, Some(&out_buf), 0);
        encoder2.set_buffer(1, Some(&partial_buf), 0);
        encoder2.set_buffer(2, Some(&out_buf), 0);
        encoder2.set_buffer(3, Some(&seq_k_buf), 0);

        let threads_per_group2 = MTLSize::new(32, 1, 1);
        let num_groups2 = MTLSize::new(1, NUM_HEADS as u64, 1);
        encoder2.dispatch_thread_groups(num_groups2, threads_per_group2);
        encoder2.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let time_us = elapsed.as_nanos() as f64 / (NUM_ROUNDS as f64 * 1000.0);
    let flops_per_head = 4.0 * seq_k as f64 * HEAD_DIM as f64;
    let total_flops = flops_per_head * NUM_HEADS as f64;
    total_flops / time_us / 1000.0
}

fn bench_llama_kernel(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
) -> f64 {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;
    let src_data: Vec<f32> = vec![0.5; num_elements];

    let (k_buf, v_buf, block_size) = match format {
        "q5_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_0(&src_data, &mut blocks);
            let size = num_blocks * 22;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 22) // BlockQ5_0 is 22 bytes per 32 elements
        }
        "q5_1" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_1> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_1(&src_data, &mut blocks);
            let size = num_blocks * 24;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 24) // BlockQ5_1 is 24 bytes per 32 elements
        }
        _ => panic!("Unknown format"),
    };

    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Get llama.cpp decode kernel
    let kernel_name = format!("kernel_flash_attn_ext_{}_dk128_dv128", format);
    let func = library.get_function(&kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    // llama.cpp kernel args structure (simplified)
    // The actual kernel uses ggml_metal_kargs_flash_attn_ext
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
        nb01_k: u32,
        nb02_k: u32,
        nb03_k: u32,
        nb01_v: u32,
        nb02_v: u32,
        nb03_v: u32,
        // Additional fields
        nb21: u32,
        nb22: u32,
        nb23: u32,
        ne0_0: u32,
        ne0_1: u32,
        ne0_2: u32,
        nb01_o: u32,
        nb02_o: u32,
        nb03_o: u32,
        pad: [u32; 4],
    }

    let row_size = HEAD_DIM / 32 * block_size;

    let args = LlamaArgs {
        ne01: HEAD_DIM as u32,                         // Q cols
        ne02: NUM_HEADS as u32,                        // Q heads
        ne03: 1,                                       // Q batch
        nb01: (HEAD_DIM * 2) as u32,                   // Q row stride (F16)
        nb02: (NUM_HEADS * HEAD_DIM * 2) as u32,       // Q head stride
        nb03: (NUM_HEADS * HEAD_DIM * 2) as u32,       // Q batch stride
        ne11: HEAD_DIM as u32,                         // K cols
        ne_12_2: NUM_HEADS as u32,                     // K heads
        ne_12_3: seq_k as u32,                         // K rows
        nb01_k: row_size as u32,                       // K row stride
        nb02_k: (seq_k * row_size) as u32,             // K head stride
        nb03_k: (NUM_HEADS * seq_k * row_size) as u32, // K batch stride
        nb01_v: row_size as u32,                       // V row stride
        nb02_v: (seq_k * row_size) as u32,             // V head stride
        nb03_v: (NUM_HEADS * seq_k * row_size) as u32, // V batch stride
        nb21: (HEAD_DIM * 4) as u32, // Output row stride (F32)
        nb22: (NUM_HEADS * HEAD_DIM * 4) as u32, // Output head stride
        nb23: (NUM_HEADS * HEAD_DIM * 4) as u32, // Output batch stride
        ne0_0: HEAD_DIM as u32,
        ne0_1: NUM_HEADS as u32,
        ne0_2: 1,
        nb01_o: (HEAD_DIM * 4) as u32,
        nb02_o: (NUM_HEADS * HEAD_DIM * 4) as u32,
        nb03_o: (NUM_HEADS * HEAD_DIM * 4) as u32,
        pad: [0; 4],
    };

    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<LlamaArgs>(),
        )
    };

    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    let scale_buf = device.new_buffer_with_data(
        &scale as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Warmup
    for _ in 0..10 {
        run_llama_kernel(
            queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            &scale_buf,
        );
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_ROUNDS {
        run_llama_kernel(
            queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            &scale_buf,
        );
    }
    let elapsed = start.elapsed();

    let time_us = elapsed.as_nanos() as f64 / (NUM_ROUNDS as f64 * 1000.0);
    let flops_per_head = 4.0 * seq_k as f64 * HEAD_DIM as f64;
    let total_flops = flops_per_head * NUM_HEADS as f64;
    total_flops / time_us / 1000.0
}

fn run_llama_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    scale_buf: &Buffer,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args.len() as u64, args.as_ptr() as *const _);
    enc.set_buffer(1, Some(q_buf), 0);
    enc.set_buffer(2, Some(k_buf), 0);
    enc.set_buffer(3, Some(v_buf), 0);
    enc.set_buffer(4, Some(out_buf), 0);
    enc.set_buffer(5, Some(scale_buf), 0);

    // llama.cpp uses different grid size based on architecture
    let grid_size = MTLSize { width: 1, height: NUM_HEADS as u64, depth: 1 };
    let tg_size = MTLSize { width: 128, height: 1, depth: 1 };
    enc.dispatch_thread_groups(grid_size, tg_size);

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}
