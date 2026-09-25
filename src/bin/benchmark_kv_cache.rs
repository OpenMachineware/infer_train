// Benchmark KV cache quantization vs llama.cpp
use infer_train::quant::kv_cache::*;
use std::time::Instant;

fn main() {
    println!("=== KV Cache Quantization Benchmark ===\n");

    let sizes = [1024, 4096, 16384, 65536];
    let runs = 10;

    println!("Format | Size  | Quant (GB/s) | Dequant (GB/s) | Size (bytes)");
    println!("--------+-------+--------------+----------------+-------------");

    for &size in &sizes {
        // F16
        benchmark_f16(size, runs);

        // BF16
        benchmark_bf16(size, runs);

        // Q8_0
        benchmark_q8_0(size, runs);

        // Q4_0
        benchmark_q4_0(size, runs);
    }
}

fn benchmark_f16(size: usize, runs: usize) {
    let input = vec![1.0f32; size];
    let mut output_f16 = vec![0u16; size];
    let mut output_f32 = vec![0.0f32; size];

    // Quantize
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_f16(&input, &mut output_f16);
    }
    let quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let quant_gbps = (size * 4) as f64 / quant_time / 1e9;

    // Dequantize
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_f16(&output_f16, &mut output_f32);
    }
    let dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let dequant_gbps = (size * 2) as f64 / dequant_time / 1e9;

    let size_bytes = size * 2;
    println!("F16    | {:5} | {:10.2}   | {:12.2}     | {} bytes",
        size, quant_gbps, dequant_gbps, size_bytes);
}

fn benchmark_bf16(size: usize, runs: usize) {
    let input = vec![1.0f32; size];
    let mut output_bf16 = vec![0u16; size];
    let mut output_f32 = vec![0.0f32; size];

    // Quantize
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_bf16(&input, &mut output_bf16);
    }
    let quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let quant_gbps = (size * 4) as f64 / quant_time / 1e9;

    // Dequantize
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_bf16(&output_bf16, &mut output_f32);
    }
    let dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let dequant_gbps = (size * 2) as f64 / dequant_time / 1e9;

    let size_bytes = size * 2;
    println!("BF16   | {:5} | {:10.2}   | {:12.2}     | {} bytes",
        size, quant_gbps, dequant_gbps, size_bytes);
}

fn benchmark_q8_0(size: usize, runs: usize) {
    assert!(size % QK == 0);
    let nb = size / QK;

    let input = vec![1.0f32; size];
    let mut output_q8 = vec![BlockQ8_0 { d: 0, qs: [0i8; QK] }; nb];
    let mut output_f32 = vec![0.0f32; size];

    // Quantize
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q8_0(&input, &mut output_q8);
    }
    let quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let quant_gbps = (size * 4) as f64 / quant_time / 1e9;

    // Dequantize
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q8_0(&output_q8, &mut output_f32);
    }
    let dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let dequant_gbps = (nb * std::mem::size_of::<BlockQ8_0>()) as f64 / dequant_time / 1e9;

    let size_bytes = nb * std::mem::size_of::<BlockQ8_0>();
    println!("Q8_0   | {:5} | {:10.2}   | {:12.2}     | {} bytes (25%)",
        size, quant_gbps, dequant_gbps, size_bytes);
}

fn benchmark_q4_0(size: usize, runs: usize) {
    assert!(size % QK == 0);
    let nb = size / QK;

    let input = vec![1.0f32; size];
    let mut output_q4 = vec![BlockQ4_0 { d: 0, qs: [0u8; QK / 2] }; nb];
    let mut output_f32 = vec![0.0f32; size];

    // Quantize
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q4_0(&input, &mut output_q4);
    }
    let quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let quant_gbps = (size * 4) as f64 / quant_time / 1e9;

    // Dequantize
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q4_0(&output_q4, &mut output_f32);
    }
    let dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let dequant_gbps = (nb * std::mem::size_of::<BlockQ4_0>()) as f64 / dequant_time / 1e9;

    let size_bytes = nb * std::mem::size_of::<BlockQ4_0>();
    println!("Q4_0   | {:5} | {:10.2}   | {:12.2}     | {} bytes (12.5%)",
        size, quant_gbps, dequant_gbps, size_bytes);
}
