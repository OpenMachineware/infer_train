// Single attention benchmark for testing one format/size at a time
// Usage: benchmark_attention_single <format> <num_heads> <head_dim> <seq_len>

use std::env;
use std::time::Instant;
use infer_train::quant::kv_cache::{BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockIQ4NL, QK};
use infer_train::ops::attention::decode_attention_f32;

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

const NUM_ITER: usize = 20;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} <format> <num_heads> <head_dim> <seq_len>", args[0]);
        eprintln!("Formats: f32, f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl");
        std::process::exit(1);
    }

    let format = args[1].as_str();
    let num_heads: usize = args[2].parse().unwrap();
    let head_dim: usize = args[3].parse().unwrap();
    let seq_len: usize = args[4].parse().unwrap();
    let scale = 1.0 / (head_dim as f32).sqrt();

    let blocks_per_row = head_dim / QK;

    // Prepare data with random values (matching llama.cpp)
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let q: Vec<f32> = (0..num_heads * head_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let mut output = vec![0.0f32; num_heads * head_dim];

    // FLOPS = 2 * num_heads * seq_len * head_dim (counting multiply-add as 1 FLOP)
    // This matches llama.cpp's FLOPS calculation
    let flops = 2.0 * num_heads as f64 * head_dim as f64 * seq_len as f64;

    match format {
        "f32" => {
            let k = vec![0.5f32; seq_len * num_heads * head_dim];
            let v = vec![0.3f32; seq_len * num_heads * head_dim];

            // Warmup
            decode_attention_f32(&q, &k, &v, &mut output, num_heads, head_dim, seq_len, scale);

            let start = Instant::now();
            for _ in 0..NUM_ITER {
                decode_attention_f32(&q, &k, &v, &mut output, num_heads, head_dim, seq_len, scale);
            }
            let elapsed = start.elapsed();
            let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
            println!("F32 Attention: {:.2} GFLOPS", gflops);
        },
        #[cfg(target_arch = "aarch64")]
        "f16" => {
            let k_f32 = vec![0.5f32; seq_len * num_heads * head_dim];
            let v_f32 = vec![0.3f32; seq_len * num_heads * head_dim];
            let q_u16: Vec<u16> = q.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            let k_u16: Vec<u16> = k_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
            let v_u16: Vec<u16> = v_f32.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();

            unsafe {
                decode_attention_f16_neon(&q_u16, &k_u16, &v_u16, &mut output, num_heads, head_dim, seq_len, scale);

                let start = Instant::now();
                for _ in 0..NUM_ITER {
                    decode_attention_f16_neon(&q_u16, &k_u16, &v_u16, &mut output, num_heads, head_dim, seq_len, scale);
                }
                let elapsed = start.elapsed();
                let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
                println!("F16 Attention: {:.2} GFLOPS", gflops);
            }
        },
        #[cfg(target_arch = "aarch64")]
        "q8_0" => {
            let num_blocks = seq_len * num_heads * blocks_per_row;
            println!("Creating {} Q8_0 blocks...", num_blocks);
            let mut k_blocks = Vec::with_capacity(num_blocks);
            let mut v_blocks = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                k_blocks.push(BlockQ8_0 { d: half::f16::from_f32(0.5).to_bits(), qs: [50i8; QK] });
                v_blocks.push(BlockQ8_0 { d: half::f16::from_f32(0.3).to_bits(), qs: [30i8; QK] });
            }
            println!("Blocks created. Running benchmark...");

            unsafe {
                decode_attention_q8_0_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);
                println!("Warmup done.");

                let start = Instant::now();
                for _ in 0..NUM_ITER {
                    decode_attention_q8_0_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);
                }
                let elapsed = start.elapsed();
                let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
                println!("Q8_0 Attention: {:.2} GFLOPS", gflops);
            }
        },
        #[cfg(target_arch = "aarch64")]
        "q4_0" => {
            let num_blocks = seq_len * num_heads * blocks_per_row;
            let mut k_blocks = Vec::with_capacity(num_blocks);
            let mut v_blocks = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                k_blocks.push(BlockQ4_0 { d: half::f16::from_f32(0.5).to_bits(), qs: [0x88u8; QK / 2] });
                v_blocks.push(BlockQ4_0 { d: half::f16::from_f32(0.3).to_bits(), qs: [0x88u8; QK / 2] });
            }

            unsafe {
                decode_attention_q4_0_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);

                let start = Instant::now();
                for _ in 0..NUM_ITER {
                    decode_attention_q4_0_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);
                }
                let elapsed = start.elapsed();
                let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
                println!("Q4_0 Attention: {:.2} GFLOPS", gflops);
            }
        },
        #[cfg(target_arch = "aarch64")]
        "q4_1" => {
            let num_blocks = seq_len * num_heads * blocks_per_row;
            let mut k_blocks = Vec::with_capacity(num_blocks);
            let mut v_blocks = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                k_blocks.push(BlockQ4_1 {
                    d: half::f16::from_f32(0.5).to_bits(),
                    m: half::f16::from_f32(0.0).to_bits(),
                    qs: [0x88u8; QK / 2]
                });
                v_blocks.push(BlockQ4_1 {
                    d: half::f16::from_f32(0.3).to_bits(),
                    m: half::f16::from_f32(0.0).to_bits(),
                    qs: [0x88u8; QK / 2]
                });
            }

            unsafe {
                decode_attention_q4_1_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);

                let start = Instant::now();
                for _ in 0..NUM_ITER {
                    decode_attention_q4_1_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);
                }
                let elapsed = start.elapsed();
                let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
                println!("Q4_1 Attention: {:.2} GFLOPS", gflops);
            }
        },
        #[cfg(target_arch = "aarch64")]
        "q5_0" => {
            let num_blocks = seq_len * num_heads * blocks_per_row;
            let mut k_blocks = Vec::with_capacity(num_blocks);
            let mut v_blocks = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                k_blocks.push(BlockQ5_0 {
                    d: half::f16::from_f32(0.5).to_bits(),
                    qh: [0u8; 4],
                    qs: [0x88u8; QK / 2]
                });
                v_blocks.push(BlockQ5_0 {
                    d: half::f16::from_f32(0.3).to_bits(),
                    qh: [0u8; 4],
                    qs: [0x88u8; QK / 2]
                });
            }

            unsafe {
                decode_attention_q5_0_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);

                let start = Instant::now();
                for _ in 0..NUM_ITER {
                    decode_attention_q5_0_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);
                }
                let elapsed = start.elapsed();
                let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
                println!("Q5_0 Attention: {:.2} GFLOPS", gflops);
            }
        },
        #[cfg(target_arch = "aarch64")]
        "q5_1" => {
            let num_blocks = seq_len * num_heads * blocks_per_row;
            let mut k_blocks = Vec::with_capacity(num_blocks);
            let mut v_blocks = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                k_blocks.push(BlockQ5_1 {
                    d: half::f16::from_f32(0.5).to_bits(),
                    m: half::f16::from_f32(0.0).to_bits(),
                    qh: [0u8; 4],
                    qs: [0x88u8; QK / 2]
                });
                v_blocks.push(BlockQ5_1 {
                    d: half::f16::from_f32(0.3).to_bits(),
                    m: half::f16::from_f32(0.0).to_bits(),
                    qh: [0u8; 4],
                    qs: [0x88u8; QK / 2]
                });
            }

            unsafe {
                decode_attention_q5_1_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);

                let start = Instant::now();
                for _ in 0..NUM_ITER {
                    decode_attention_q5_1_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);
                }
                let elapsed = start.elapsed();
                let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
                println!("Q5_1 Attention: {:.2} GFLOPS", gflops);
            }
        },
        #[cfg(target_arch = "aarch64")]
        "iq4_nl" => {
            let num_blocks = seq_len * num_heads * blocks_per_row;
            let mut k_blocks = Vec::with_capacity(num_blocks);
            let mut v_blocks = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                k_blocks.push(BlockIQ4NL {
                    d: half::f16::from_f32(0.5).to_bits(),
                    qs: [0x88u8; QK / 2]
                });
                v_blocks.push(BlockIQ4NL {
                    d: half::f16::from_f32(0.3).to_bits(),
                    qs: [0x88u8; QK / 2]
                });
            }

            unsafe {
                decode_attention_iq4_nl_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);

                let start = Instant::now();
                for _ in 0..NUM_ITER {
                    decode_attention_iq4_nl_neon(&q, &k_blocks, &v_blocks, &mut output, num_heads, head_dim, seq_len, scale);
                }
                let elapsed = start.elapsed();
                let gflops = (flops * NUM_ITER as f64) / elapsed.as_secs_f64() / 1e9;
                println!("IQ4_NL Attention: {:.2} GFLOPS", gflops);
            }
        },
        _ => {
            eprintln!("Unknown format: {}", format);
            std::process::exit(1);
        }
    }
}
