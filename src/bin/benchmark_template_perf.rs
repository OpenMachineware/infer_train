// Benchmark attention operator vs llama.cpp
// Tests all 9 KV cache formats with fused NEON implementation

use std::time::Instant;
use infer_train::quant::kv_cache::{
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockIQ4NL, QK,
};
use infer_train::ops::attention::{
    decode_attention_f32,
};
#[cfg(target_arch = "aarch64")]
use infer_train::ops::attention::{
    decode_attention_f16_neon,
    decode_attention_q8_0_neon,
    decode_attention_q4_0_neon,
    decode_attention_q4_1_neon,
    decode_attention_q5_0_neon,
    decode_attention_q5_1_neon,
    decode_attention_iq4_nl_neon,
};

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ITER: usize = 10;

fn benchmark_fp16(seq_len: usize) {
    let q_f32 = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let k_f32 = vec![0.5f32; seq_len * NUM_HEADS * HEAD_DIM];
    let v_f32 = vec![0.3f32; seq_len * NUM_HEADS * HEAD_DIM];
    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Convert to FP16
    let q: Vec<u16> = q_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    let k: Vec<u16> = k_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    let v: Vec<u16> = v_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        // Warmup
        decode_attention_f16_neon(&q, &k, &v, &mut output, NUM_HEADS, HEAD_DIM, seq_len, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_f16_neon(&q, &k, &v, &mut output, NUM_HEADS, HEAD_DIM, seq_len, scale);
        }
        let elapsed = start.elapsed();

        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * seq_len as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;

        println!("f16\t| {:8.2} GFLOPS | {:.3} ms/iter", gflops, elapsed.as_secs_f64() * 1000.0 / NUM_ITER as f64);
    }

    #[cfg(not(target_arch = "aarch64"))]
    println!("FP16 benchmark only available on aarch64");
}

fn benchmark_q8_0(seq_len: usize) {
    let q = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let blocks_per_row = HEAD_DIM / QK;
    let num_k_blocks = seq_len * NUM_HEADS * blocks_per_row;

    let mut k_blocks: Vec<BlockQ8_0> = Vec::with_capacity(num_k_blocks);
    let mut v_blocks: Vec<BlockQ8_0> = Vec::with_capacity(num_k_blocks);

    for _ in 0..num_k_blocks {
        let scale = half::f16::from_f32(0.5).to_bits();
        let qs = [50i8; QK];
        k_blocks.push(BlockQ8_0 { d: scale, qs });
        v_blocks.push(BlockQ8_0 { d: scale, qs });
    }

    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        // Warmup
        decode_attention_q8_0_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, seq_len, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_q8_0_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, seq_len, scale);
        }
        let elapsed = start.elapsed();

        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * seq_len as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;

        println!("q8_0\t| {:8.2} GFLOPS | {:.3} ms/iter", gflops, elapsed.as_secs_f64() * 1000.0 / NUM_ITER as f64);
    }

    #[cfg(not(target_arch = "aarch64"))]
    println!("Q8_0 benchmark only available on aarch64");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let seq_len = if args.len() > 1 {
        args[1].parse().unwrap_or(512)
    } else {
        512
    };

    println!("=== Attention Benchmark ===");
    println!("Config: {} heads, {} head_dim, {} seq_len, {} iterations", NUM_HEADS, HEAD_DIM, seq_len, NUM_ITER);
    println!();

    benchmark_fp16(seq_len);
    benchmark_q8_0(seq_len);
}
