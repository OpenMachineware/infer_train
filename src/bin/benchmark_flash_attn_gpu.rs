// Benchmark GPU Flash Attention vs llama.cpp
// Tests decode case (batch=1) with different seq_len and formats

use std::time::Instant;
use infer_train::ops::gpu::metal::{FlashAttnMetalContext, KvFormat};
use infer_train::quant::kv_cache::{BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, QK};

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ITER: usize = 50;

fn create_f16_data(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<u16> {
    let len = seq_len * n_heads * head_dim;
    (0..len).map(|i| half::f16::from_f32(0.5 * (i as f32 % 100.0) / 100.0).to_bits()).collect()
}

fn create_q4_0_data(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<BlockQ4_0> {
    let blocks_per_row = head_dim / QK;
    let num_blocks = seq_len * n_heads * blocks_per_row;
    (0..num_blocks).map(|_| BlockQ4_0 {
        d: half::f16::from_f32(0.5).to_bits(),
        qs: [0x88u8; QK / 2],
    }).collect()
}

fn create_q4_1_data(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<BlockQ4_1> {
    let blocks_per_row = head_dim / QK;
    let num_blocks = seq_len * n_heads * blocks_per_row;
    (0..num_blocks).map(|_| BlockQ4_1 {
        d: half::f16::from_f32(0.5).to_bits(),
        m: half::f16::from_f32(0.0).to_bits(),
        qs: [0x88u8; QK / 2],
    }).collect()
}

fn create_q5_0_data(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<BlockQ5_0> {
    let blocks_per_row = head_dim / QK;
    let num_blocks = seq_len * n_heads * blocks_per_row;
    (0..num_blocks).map(|_| BlockQ5_0 {
        d: half::f16::from_f32(0.5).to_bits(),
        qh: [0u8; 4],
        qs: [0x88u8; QK / 2],
    }).collect()
}

fn create_q5_1_data(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<BlockQ5_1> {
    let blocks_per_row = head_dim / QK;
    let num_blocks = seq_len * n_heads * blocks_per_row;
    (0..num_blocks).map(|_| BlockQ5_1 {
        d: half::f16::from_f32(0.5).to_bits(),
        m: half::f16::from_f32(0.0).to_bits(),
        qh: [0u8; 4],
        qs: [0x88u8; QK / 2],
    }).collect()
}

fn create_q8_0_data(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<BlockQ8_0> {
    let blocks_per_row = head_dim / QK;
    let num_blocks = seq_len * n_heads * blocks_per_row;
    (0..num_blocks).map(|_| BlockQ8_0 {
        d: half::f16::from_f32(0.5).to_bits(),
        qs: [50i8; QK],
    }).collect()
}

fn benchmark_format(
    ctx: &FlashAttnMetalContext,
    format: KvFormat,
    seq_len: usize,
    q: &[f32],
    k_bytes: &[u8],
    v_bytes: &[u8],
) -> Result<f64, String> {
    // Warmup
    let _ = ctx.flash_attn_decode(
        q, k_bytes, v_bytes, None,
        NUM_HEADS as u32, NUM_HEADS as u32, HEAD_DIM as u32, seq_len as u32, format,
    )?;

    let start = Instant::now();
    for _ in 0..NUM_ITER {
        let _ = ctx.flash_attn_decode(
            q, k_bytes, v_bytes, None,
            NUM_HEADS as u32, NUM_HEADS as u32, HEAD_DIM as u32, seq_len as u32, format,
        )?;
    }
    let elapsed = start.elapsed();

    // FLOPs: 2 * n_heads * head_dim * seq_len (Q@K + V@attn)
    let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * seq_len as f64;
    let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;

    Ok(gflops)
}

fn main() {
    println!("=== GPU Flash Attention Benchmark ===");
    println!("Config: {} heads, {} head_dim", NUM_HEADS, HEAD_DIM);
    println!();

    let ctx = match FlashAttnMetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    // Create query (FP32)
    let q: Vec<f32> = vec![1.0; NUM_HEADS * HEAD_DIM];

    // Test different seq_len
    let seq_lens = [128, 256, 512, 1024, 2048, 4096];

    println!("{:<10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "seq_len", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0");
    println!("{}", "-".repeat(82));

    for seq_len in seq_lens {
        // Create test data for each format
        let k_f16 = create_f16_data(seq_len, NUM_HEADS, HEAD_DIM);
        let v_f16 = create_f16_data(seq_len, NUM_HEADS, HEAD_DIM);

        let k_q4_0 = create_q4_0_data(seq_len, NUM_HEADS, HEAD_DIM);
        let v_q4_0 = create_q4_0_data(seq_len, NUM_HEADS, HEAD_DIM);

        let k_q4_1 = create_q4_1_data(seq_len, NUM_HEADS, HEAD_DIM);
        let v_q4_1 = create_q4_1_data(seq_len, NUM_HEADS, HEAD_DIM);

        let k_q5_0 = create_q5_0_data(seq_len, NUM_HEADS, HEAD_DIM);
        let v_q5_0 = create_q5_0_data(seq_len, NUM_HEADS, HEAD_DIM);

        let k_q5_1 = create_q5_1_data(seq_len, NUM_HEADS, HEAD_DIM);
        let v_q5_1 = create_q5_1_data(seq_len, NUM_HEADS, HEAD_DIM);

        let k_q8_0 = create_q8_0_data(seq_len, NUM_HEADS, HEAD_DIM);
        let v_q8_0 = create_q8_0_data(seq_len, NUM_HEADS, HEAD_DIM);

        // Benchmark each format
        let f16_gflops = benchmark_format(&ctx, KvFormat::F16, seq_len, &q,
            unsafe { std::slice::from_raw_parts(k_f16.as_ptr() as *const u8, k_f16.len() * 2) },
            unsafe { std::slice::from_raw_parts(v_f16.as_ptr() as *const u8, v_f16.len() * 2) }
        ).unwrap_or(0.0);

        let q4_0_gflops = benchmark_format(&ctx, KvFormat::Q4_0, seq_len, &q,
            unsafe { std::slice::from_raw_parts(k_q4_0.as_ptr() as *const u8, k_q4_0.len() * std::mem::size_of::<BlockQ4_0>()) },
            unsafe { std::slice::from_raw_parts(v_q4_0.as_ptr() as *const u8, v_q4_0.len() * std::mem::size_of::<BlockQ4_0>()) }
        ).unwrap_or(0.0);

        let q4_1_gflops = benchmark_format(&ctx, KvFormat::Q4_1, seq_len, &q,
            unsafe { std::slice::from_raw_parts(k_q4_1.as_ptr() as *const u8, k_q4_1.len() * std::mem::size_of::<BlockQ4_1>()) },
            unsafe { std::slice::from_raw_parts(v_q4_1.as_ptr() as *const u8, v_q4_1.len() * std::mem::size_of::<BlockQ4_1>()) }
        ).unwrap_or(0.0);

        let q5_0_gflops = benchmark_format(&ctx, KvFormat::Q5_0, seq_len, &q,
            unsafe { std::slice::from_raw_parts(k_q5_0.as_ptr() as *const u8, k_q5_0.len() * std::mem::size_of::<BlockQ5_0>()) },
            unsafe { std::slice::from_raw_parts(v_q5_0.as_ptr() as *const u8, v_q5_0.len() * std::mem::size_of::<BlockQ5_0>()) }
        ).unwrap_or(0.0);

        let q5_1_gflops = benchmark_format(&ctx, KvFormat::Q5_1, seq_len, &q,
            unsafe { std::slice::from_raw_parts(k_q5_1.as_ptr() as *const u8, k_q5_1.len() * std::mem::size_of::<BlockQ5_1>()) },
            unsafe { std::slice::from_raw_parts(v_q5_1.as_ptr() as *const u8, v_q5_1.len() * std::mem::size_of::<BlockQ5_1>()) }
        ).unwrap_or(0.0);

        let q8_0_gflops = benchmark_format(&ctx, KvFormat::Q8_0, seq_len, &q,
            unsafe { std::slice::from_raw_parts(k_q8_0.as_ptr() as *const u8, k_q8_0.len() * std::mem::size_of::<BlockQ8_0>()) },
            unsafe { std::slice::from_raw_parts(v_q8_0.as_ptr() as *const u8, v_q8_0.len() * std::mem::size_of::<BlockQ8_0>()) }
        ).unwrap_or(0.0);

        println!("{:<10} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            seq_len, f16_gflops, q4_0_gflops, q4_1_gflops, q5_0_gflops, q5_1_gflops, q8_0_gflops);
    }

    println!();
    println!("Note: GFLOPS = 2 * n_heads * head_dim * seq_len / time");
    println!("      Higher is better");
}