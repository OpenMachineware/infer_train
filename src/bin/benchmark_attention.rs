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
    decode_attention_bf16_neon,
    decode_attention_q8_0_neon,
    decode_attention_q4_0_neon,
    decode_attention_q4_1_neon,
    decode_attention_q5_0_neon,
    decode_attention_q5_1_neon,
    decode_attention_iq4_nl_neon,
};

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const SEQ_LEN: usize = 512;
const NUM_ITER: usize = 50;

fn benchmark_fp32() {
    let q = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let k = vec![0.5f32; SEQ_LEN * NUM_HEADS * HEAD_DIM];
    let v = vec![0.3f32; SEQ_LEN * NUM_HEADS * HEAD_DIM];
    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Warmup
    decode_attention_f32(&q, &k, &v, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);

    let start = Instant::now();
    for _ in 0..NUM_ITER {
        decode_attention_f32(&q, &k, &v, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);
    }
    let elapsed = start.elapsed();

    let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * SEQ_LEN as f64; // Q@K + V@attn
    let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;

    println!("FP32 Attention: {:.2} GFLOPS ({:.2?} per iter)", gflops, elapsed / NUM_ITER as u32);
}

#[cfg(target_arch = "aarch64")]
fn benchmark_fp16() {
    let q_f32 = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let k_f32 = vec![0.5f32; SEQ_LEN * NUM_HEADS * HEAD_DIM];
    let v_f32 = vec![0.3f32; SEQ_LEN * NUM_HEADS * HEAD_DIM];
    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Convert to FP16
    let q: Vec<u16> = q_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    let k: Vec<u16> = k_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    let v: Vec<u16> = v_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();

    unsafe {
        // Warmup
        decode_attention_f16_neon(&q, &k, &v, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_f16_neon(&q, &k, &v, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);
        }
        let elapsed = start.elapsed();

        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * SEQ_LEN as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;

        println!("FP16 Attention: {:.2} GFLOPS ({:.2?} per iter)", gflops, elapsed / NUM_ITER as u32);
    }
}

#[cfg(target_arch = "aarch64")]
fn benchmark_q8_0() {
    let q = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let blocks_per_row = HEAD_DIM / QK;
    let num_k_blocks = SEQ_LEN * NUM_HEADS * blocks_per_row;

    // Create quantized K/V
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

    unsafe {
        // Warmup
        decode_attention_q8_0_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_q8_0_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);
        }
        let elapsed = start.elapsed();

        // Q8_0: 2 bytes scale + 32 bytes qs = 34 bytes per 32 floats = 1.0625 bytes/float
        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * SEQ_LEN as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
        let bandwidth = (NUM_HEADS * SEQ_LEN * HEAD_DIM * 2 * 34 / 32) as f64 * NUM_ITER as f64
            / elapsed.as_secs_f64() / 1e9;

        println!("Q8_0 Attention: {:.2} GFLOPS, {:.2} GB/s ({:.2?} per iter)", gflops, bandwidth, elapsed / NUM_ITER as u32);
    }
}

#[cfg(target_arch = "aarch64")]
fn benchmark_q4_0() {
    let q = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let blocks_per_row = HEAD_DIM / QK;
    let num_k_blocks = SEQ_LEN * NUM_HEADS * blocks_per_row;

    // Create quantized K/V
    let mut k_blocks: Vec<BlockQ4_0> = Vec::with_capacity(num_k_blocks);
    let mut v_blocks: Vec<BlockQ4_0> = Vec::with_capacity(num_k_blocks);

    for _ in 0..num_k_blocks {
        let scale = half::f16::from_f32(0.5).to_bits();
        let qs = [0x88u8; QK / 2]; // nibbles representing 8 (zero after bias)
        k_blocks.push(BlockQ4_0 { d: scale, qs: qs.clone() });
        v_blocks.push(BlockQ4_0 { d: scale, qs });
    }

    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    unsafe {
        // Warmup
        decode_attention_q4_0_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_q4_0_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);
        }
        let elapsed = start.elapsed();

        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * SEQ_LEN as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
        // Q4_0: 2 bytes scale + 16 bytes qs = 18 bytes per 32 floats = 0.5625 bytes/float
        let bandwidth = (NUM_HEADS * SEQ_LEN * HEAD_DIM * 2 * 18 / 32) as f64 * NUM_ITER as f64
            / elapsed.as_secs_f64() / 1e9;

        println!("Q4_0 Attention: {:.2} GFLOPS, {:.2} GB/s ({:.2?} per iter)", gflops, bandwidth, elapsed / NUM_ITER as u32);
    }
}

#[cfg(target_arch = "aarch64")]
fn benchmark_q4_1() {
    let q = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let blocks_per_row = HEAD_DIM / QK;
    let num_k_blocks = SEQ_LEN * NUM_HEADS * blocks_per_row;

    let mut k_blocks: Vec<BlockQ4_1> = Vec::with_capacity(num_k_blocks);
    let mut v_blocks: Vec<BlockQ4_1> = Vec::with_capacity(num_k_blocks);

    for _ in 0..num_k_blocks {
        let scale = half::f16::from_f32(0.5).to_bits();
        let min = half::f16::from_f32(0.0).to_bits();
        let qs = [0x88u8; QK / 2];
        k_blocks.push(BlockQ4_1 { d: scale, m: min, qs: qs.clone() });
        v_blocks.push(BlockQ4_1 { d: scale, m: min, qs });
    }

    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    unsafe {
        decode_attention_q4_1_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_q4_1_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);
        }
        let elapsed = start.elapsed();

        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * SEQ_LEN as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
        // Q4_1: 4 bytes scales + 16 bytes qs = 20 bytes per 32 floats
        let bandwidth = (NUM_HEADS * SEQ_LEN * HEAD_DIM * 2 * 20 / 32) as f64 * NUM_ITER as f64
            / elapsed.as_secs_f64() / 1e9;

        println!("Q4_1 Attention: {:.2} GFLOPS, {:.2} GB/s ({:.2?} per iter)", gflops, bandwidth, elapsed / NUM_ITER as u32);
    }
}

#[cfg(target_arch = "aarch64")]
fn benchmark_q5_0() {
    let q = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let blocks_per_row = HEAD_DIM / QK;
    let num_k_blocks = SEQ_LEN * NUM_HEADS * blocks_per_row;

    let mut k_blocks: Vec<BlockQ5_0> = Vec::with_capacity(num_k_blocks);
    let mut v_blocks: Vec<BlockQ5_0> = Vec::with_capacity(num_k_blocks);

    for _ in 0..num_k_blocks {
        let scale = half::f16::from_f32(0.5).to_bits();
        let qh = [0u8; 4];
        let qs = [0x88u8; QK / 2];
        k_blocks.push(BlockQ5_0 { d: scale, qh, qs: qs.clone() });
        v_blocks.push(BlockQ5_0 { d: scale, qh, qs });
    }

    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    unsafe {
        decode_attention_q5_0_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_q5_0_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);
        }
        let elapsed = start.elapsed();

        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * SEQ_LEN as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
        // Q5_0: 2 + 4 + 16 = 22 bytes per 32 floats
        let bandwidth = (NUM_HEADS * SEQ_LEN * HEAD_DIM * 2 * 22 / 32) as f64 * NUM_ITER as f64
            / elapsed.as_secs_f64() / 1e9;

        println!("Q5_0 Attention: {:.2} GFLOPS, {:.2} GB/s ({:.2?} per iter)", gflops, bandwidth, elapsed / NUM_ITER as u32);
    }
}

#[cfg(target_arch = "aarch64")]
fn benchmark_q5_1() {
    let q = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let blocks_per_row = HEAD_DIM / QK;
    let num_k_blocks = SEQ_LEN * NUM_HEADS * blocks_per_row;

    let mut k_blocks: Vec<BlockQ5_1> = Vec::with_capacity(num_k_blocks);
    let mut v_blocks: Vec<BlockQ5_1> = Vec::with_capacity(num_k_blocks);

    for _ in 0..num_k_blocks {
        let scale = half::f16::from_f32(0.5).to_bits();
        let min = half::f16::from_f32(0.0).to_bits();
        let qh = [0u8; 4];
        let qs = [0x88u8; QK / 2];
        k_blocks.push(BlockQ5_1 { d: scale, m: min, qh, qs: qs.clone() });
        v_blocks.push(BlockQ5_1 { d: scale, m: min, qh, qs });
    }

    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    unsafe {
        decode_attention_q5_1_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_q5_1_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);
        }
        let elapsed = start.elapsed();

        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * SEQ_LEN as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
        // Q5_1: 4 + 4 + 16 = 24 bytes per 32 floats
        let bandwidth = (NUM_HEADS * SEQ_LEN * HEAD_DIM * 2 * 24 / 32) as f64 * NUM_ITER as f64
            / elapsed.as_secs_f64() / 1e9;

        println!("Q5_1 Attention: {:.2} GFLOPS, {:.2} GB/s ({:.2?} per iter)", gflops, bandwidth, elapsed / NUM_ITER as u32);
    }
}

#[cfg(target_arch = "aarch64")]
fn benchmark_iq4_nl() {
    let q = vec![1.0f32; NUM_HEADS * HEAD_DIM];
    let blocks_per_row = HEAD_DIM / QK;
    let num_k_blocks = SEQ_LEN * NUM_HEADS * blocks_per_row;

    let mut k_blocks: Vec<BlockIQ4NL> = Vec::with_capacity(num_k_blocks);
    let mut v_blocks: Vec<BlockIQ4NL> = Vec::with_capacity(num_k_blocks);

    for _ in 0..num_k_blocks {
        let scale = half::f16::from_f32(0.5).to_bits();
        let qs = [0x88u8; QK / 2]; // indices into lookup table
        k_blocks.push(BlockIQ4NL { d: scale, qs: qs.clone() });
        v_blocks.push(BlockIQ4NL { d: scale, qs });
    }

    let mut output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    unsafe {
        decode_attention_iq4_nl_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);

        let start = Instant::now();
        for _ in 0..NUM_ITER {
            decode_attention_iq4_nl_neon(&q, &k_blocks, &v_blocks, &mut output, NUM_HEADS, HEAD_DIM, SEQ_LEN, scale);
        }
        let elapsed = start.elapsed();

        let flops = 2.0 * NUM_HEADS as f64 * HEAD_DIM as f64 * SEQ_LEN as f64;
        let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
        // IQ4_NL: 2 + 16 = 18 bytes per 32 floats
        let bandwidth = (NUM_HEADS * SEQ_LEN * HEAD_DIM * 2 * 18 / 32) as f64 * NUM_ITER as f64
            / elapsed.as_secs_f64() / 1e9;

        println!("IQ4_NL Attention: {:.2} GFLOPS, {:.2} GB/s ({:.2?} per iter)", gflops, bandwidth, elapsed / NUM_ITER as u32);
    }
}

fn main() {
    println!("=== Attention Benchmark ===");
    println!("Config: {} heads, {} head_dim, {} seq_len", NUM_HEADS, HEAD_DIM, SEQ_LEN);
    println!();

    benchmark_fp32();

    #[cfg(target_arch = "aarch64")]
    {
        benchmark_fp16();
        benchmark_q8_0();
        benchmark_q4_0();
        benchmark_q4_1();
        benchmark_q5_0();
        benchmark_q5_1();
        benchmark_iq4_nl();
    }

    #[cfg(not(target_arch = "aarch64"))]
    println!("NEON benchmarks only available on aarch64");
}
