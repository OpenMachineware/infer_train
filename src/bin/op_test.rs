// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// 通用算子测试框架 - Rust版本
// 用法:
//   1. 看输出: cargo run --release --bin op_test -- --op rms_norm --verbose
//   2. 测性能: cargo run --release --bin op_test -- --op rms_norm --bench 1000
//   3. 对比输出: cargo run --release --bin op_test -- --op vec_dot_q5_k --verbose

use std::time::Instant;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 算子名称: rms_norm, silu, vec_dot_q5_k, vec_dot_q6_k, vec_dot_q4_k, quantize_q8_k
    #[arg(short, long)]
    op: String,

    /// 打印详细输出
    #[arg(short, long)]
    verbose: bool,

    /// 性能测试循环次数
    #[arg(short, long, default_value = "0")]
    bench: u32,

    /// 输入大小 (对于需要size的算子)
    #[arg(short, long, default_value = "1024")]
    size: usize,

    /// 种子 (用于生成随机输入)
    #[arg(short, long, default_value = "42")]
    seed: u64,
}

fn generate_input(size: usize, seed: u64) -> Vec<f32> {
    // 简单的伪随机生成
    let mut state = seed;
    let mut result = Vec::with_capacity(size);
    for _ in 0..size {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let val = ((state >> 16) as i32 as f32) / 32768.0; // -1.0 to 1.0
        result.push(val);
    }
    result
}

fn test_rms_norm(args: &Args) {
    use infer_train::backend::arm::rms_norm::rms_norm_f32;

    let size = args.size;
    let input = generate_input(size, args.seed);
    let weight = generate_input(size, args.seed + 1);
    let mut output = vec![0.0f32; size];

    // 计算参考值
    let eps = 1e-5f32;
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / size as f32 + eps).sqrt();
    let scale = 1.0 / rms;

    if args.verbose {
        println!("=== RMS Norm ===");
        println!("Size: {}", size);
        println!("Input RMS: {:.6}", rms);
        println!("Scale: {:.6}", scale);
        println!("Input [0..5]: {:?}", &input[..5.min(size)]);
        println!("Weight [0..5]: {:?}", &weight[..5.min(size)]);
    }

    // 单次执行看输出
    rms_norm_f32(&input, &weight, &mut output, eps);

    if args.verbose {
        println!("Output [0..5]: {:?}", &output[..5.min(size)]);
        let out_min = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let out_max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("Output range: [{:.4}, {:.4}]", out_min, out_max);

        // 验证输出RMS ≈ 1.0 (当weight=1时)
        let out_sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let out_rms = (out_sum_sq / size as f32).sqrt();
        println!("Output RMS (should be ~1.0 with weight=1): {:.4}", out_rms);
    }

    // 性能测试
    if args.bench > 0 {
        let start = Instant::now();
        for _ in 0..args.bench {
            rms_norm_f32(&input, &weight, &mut output, eps);
        }
        let elapsed = start.elapsed();
        let us_per_iter = elapsed.as_micros() as f64 / args.bench as f64;
        println!("Benchmark: {} iterations in {:?}", args.bench, elapsed);
        println!("Per iteration: {:.2} µs ({:.2} GB/s)", us_per_iter,
                 (size as f64 * 4.0 * 3.0) / us_per_iter / 1000.0);
    }
}

fn test_silu(args: &Args) {
    use infer_train::backend::arm::activation::silu_f32;

    let size = args.size;
    let input = generate_input(size, args.seed);
    let mut output = vec![0.0f32; size];

    if args.verbose {
        println!("=== SiLU Activation ===");
        println!("Size: {}", size);
        println!("Input [0..5]: {:?}", &input[..5.min(size)]);
    }

    silu_f32(&input, &mut output);

    if args.verbose {
        println!("Output [0..5]: {:?}", &output[..5.min(size)]);
        let out_min = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let out_max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("Output range: [{:.4}, {:.4}]", out_min, out_max);
    }

    if args.bench > 0 {
        let start = Instant::now();
        for _ in 0..args.bench {
            silu_f32(&input, &mut output);
        }
        let elapsed = start.elapsed();
        let us_per_iter = elapsed.as_micros() as f64 / args.bench as f64;
        println!("Benchmark: {} iterations in {:?}", args.bench, elapsed);
        println!("Per iteration: {:.2} µs", us_per_iter);
    }
}

fn test_quantize_q8_k(args: &Args) {
    use infer_train::types::{BlockQ8_K, QK_K};
    use infer_train::backend::arm::quant::quantize_q8_k;

    let size = args.size;
    let nb = (size + QK_K - 1) / QK_K;
    let input = generate_input(size, args.seed);
    // 放大范围
    let input: Vec<f32> = input.iter().map(|&x| x * 10.0).collect();
    let mut output: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];

    if args.verbose {
        println!("=== Quantize Q8_K ===");
        println!("Size: {} ({} blocks)", size, nb);
        let in_min = input.iter().cloned().fold(f32::INFINITY, f32::min);
        let in_max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("Input range: [{:.4}, {:.4}]", in_min, in_max);
        println!("Input [0..5]: {:?}", &input[..5.min(size)]);
    }

    quantize_q8_k(&input, &mut output);

    if args.verbose {
        println!("\nFirst block:");
        println!("  d = {:.6}", output[0].d);
        println!("  qs [0..10]: {:?}", &output[0].qs[..10.min(256)]);
        println!("  bsums [0..4]: {:?}", &output[0].bsums[..4.min(16)]);

        // 验证: dequant = d * qs
        println!("\nVerification:");
        for i in 0..5.min(256) {
            let deq = output[0].d * (output[0].qs[i] as f32);
            println!("  qs[{}] = {}, dequant = {:.4}, original = {:.4}",
                     i, output[0].qs[i], deq, input[i]);
        }
    }

    if args.bench > 0 {
        let start = Instant::now();
        for _ in 0..args.bench {
            quantize_q8_k(&input, &mut output);
        }
        let elapsed = start.elapsed();
        let us_per_iter = elapsed.as_micros() as f64 / args.bench as f64;
        println!("Benchmark: {} iterations in {:?}", args.bench, elapsed);
        println!("Per iteration: {:.2} µs ({:.2} GB/s)", us_per_iter,
                 (size as f64 * 4.0) / us_per_iter / 1000.0);
    }
}

fn test_vec_dot_q5_k(args: &Args) {
    use infer_train::types::{BlockQ5_K, BlockQ8_K, QK_K, fp16_to_fp32};
    use infer_train::backend::arm::{quant::quantize_q8_k, vec_dot::vec_dot_q5_k_q8_k};

    // 创建测试输入
    let input_size = QK_K; // 256
    let nb = 4; // 4 blocks = 1024 elements
    let total_size = input_size * nb;

    let input = generate_input(total_size, args.seed);
    let input: Vec<f32> = input.iter().map(|&x| x * 5.0).collect(); // 放大范围

    // 量化输入到Q8_K
    let mut input_q8k: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_q8_k(&input, &mut input_q8k);

    // 创建假的Q5_K权重 (需要正确的格式)
    let mut weights: Vec<BlockQ5_K> = vec![unsafe { std::mem::zeroed() }; nb];
    // 填充一些值
    for b in &mut weights {
        b.d = half::f16::to_bits(half::f16::from_f32(0.001)); // scale
        b.dmin = half::f16::to_bits(half::f16::from_f32(0.01)); // min scale
        // qs: 低4位和高4位各存储一个值
        for i in 0..128 {
            b.qs[i] = 0x88; // 低4位=8, 高4位=8
        }
        // qh: 高位存储
        for i in 0..32 {
            b.qh[i] = 0;
        }
        // scales: 子块scale
        for i in 0..12 {
            b.scales[i] = 32; // 中等scale
        }
    }

    if args.verbose {
        println!("=== vec_dot Q5_K x Q8_K ===");
        println!("Input size: {} ({} blocks)", total_size, nb);
        println!("Input [0..5]: {:?}", &input[..5]);
        println!("Q8_K block 0: d = {:.6}", input_q8k[0].d);
        println!("Q8_K block 0 qs [0..5]: {:?}", &input_q8k[0].qs[..5]);
        println!("Q5_K block 0: d = {:.6}, dmin = {:.6}", fp16_to_fp32(weights[0].d), fp16_to_fp32(weights[0].dmin));
    }

    // 计算点积
    let result = vec_dot_q5_k_q8_k(&weights, &input_q8k);

    if args.verbose {
        println!("\nResult: {:.6}", result);
    }

    if args.bench > 0 {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..args.bench {
            sum += vec_dot_q5_k_q8_k(&weights, &input_q8k);
        }
        let elapsed = start.elapsed();
        let us_per_iter = elapsed.as_micros() as f64 / args.bench as f64;
        println!("Benchmark: {} iterations in {:?}", args.bench, elapsed);
        println!("Per iteration: {:.2} µs ({:.2} GOPS)", us_per_iter,
                 (total_size as f64 * 2.0) / us_per_iter / 1000.0);
        println!("(sum = {} to prevent optimization)", sum);
    }
}

fn test_vec_dot_q6_k(args: &Args) {
    use infer_train::types::{BlockQ6_K, BlockQ8_K, QK_K, fp16_to_fp32};
    use infer_train::backend::arm::{quant::quantize_q8_k, vec_dot::vec_dot_q6_k_q8_k};

    let input_size = QK_K;
    let nb = 4;
    let total_size = input_size * nb;

    let input = generate_input(total_size, args.seed);
    let input: Vec<f32> = input.iter().map(|&x| x * 5.0).collect();

    let mut input_q8k: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_q8_k(&input, &mut input_q8k);

    // 创建假的Q6_K权重
    let mut weights: Vec<BlockQ6_K> = vec![unsafe { std::mem::zeroed() }; nb];
    for b in &mut weights {
        b.d = half::f16::to_bits(half::f16::from_f32(0.001));
        for i in 0..128 {
            b.ql[i] = 32; // 中等值
        }
        for i in 0..64 {
            b.qh[i] = 0;
        }
    }

    if args.verbose {
        println!("=== vec_dot Q6_K x Q8_K ===");
        println!("Input size: {} ({} blocks)", total_size, nb);
        println!("Q8_K block 0: d = {:.6}", input_q8k[0].d);
        println!("Q6_K block 0: d = {:.6}", fp16_to_fp32(weights[0].d));
    }

    let result = vec_dot_q6_k_q8_k(&weights, &input_q8k);

    if args.verbose {
        println!("Result: {:.6}", result);
    }

    if args.bench > 0 {
        let start = Instant::now();
        let mut _sum = 0.0f32;
        for _ in 0..args.bench {
            _sum += vec_dot_q6_k_q8_k(&weights, &input_q8k);
        }
        let elapsed = start.elapsed();
        let us_per_iter = elapsed.as_micros() as f64 / args.bench as f64;
        println!("Benchmark: {} iterations in {:?}", args.bench, elapsed);
        println!("Per iteration: {:.2} µs", us_per_iter);
    }
}

fn test_vec_dot_q4_k(args: &Args) {
    use infer_train::types::{BlockQ4_K, BlockQ8_K, QK_K, fp16_to_fp32};
    use infer_train::backend::arm::{quant::quantize_q8_k, vec_dot::vec_dot_q4_k_q8_k};

    let input_size = QK_K;
    let nb = 4;
    let total_size = input_size * nb;

    let input = generate_input(total_size, args.seed);
    let input: Vec<f32> = input.iter().map(|&x| x * 5.0).collect();

    let mut input_q8k: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_q8_k(&input, &mut input_q8k);

    // 创建假的Q4_K权重
    let mut weights: Vec<BlockQ4_K> = vec![unsafe { std::mem::zeroed() }; nb];
    for b in &mut weights {
        b.d = half::f16::to_bits(half::f16::from_f32(0.001));
        b.dmin = half::f16::to_bits(half::f16::from_f32(0.01));
        for i in 0..128 {
            b.qs[i] = 0x88;
        }
        for i in 0..12 {
            b.scales[i] = 32;
        }
    }

    if args.verbose {
        println!("=== vec_dot Q4_K x Q8_K ===");
        println!("Input size: {} ({} blocks)", total_size, nb);
        println!("Q8_K block 0: d = {:.6}", input_q8k[0].d);
        println!("Q4_K block 0: d = {:.6}", fp16_to_fp32(weights[0].d));
    }

    let result = vec_dot_q4_k_q8_k(&weights, &input_q8k);

    if args.verbose {
        println!("Result: {:.6}", result);
    }

    if args.bench > 0 {
        let start = Instant::now();
        let mut _sum = 0.0f32;
        for _ in 0..args.bench {
            _sum += vec_dot_q4_k_q8_k(&weights, &input_q8k);
        }
        let elapsed = start.elapsed();
        let us_per_iter = elapsed.as_micros() as f64 / args.bench as f64;
        println!("Benchmark: {} iterations in {:?}", args.bench, elapsed);
        println!("Per iteration: {:.2} µs", us_per_iter);
    }
}

fn main() {
    let args = Args::parse();

    match args.op.as_str() {
        "rms_norm" => test_rms_norm(&args),
        "silu" => test_silu(&args),
        "quantize_q8_k" => test_quantize_q8_k(&args),
        "vec_dot_q5_k" => test_vec_dot_q5_k(&args),
        "vec_dot_q6_k" => test_vec_dot_q6_k(&args),
        "vec_dot_q4_k" => test_vec_dot_q4_k(&args),
        _ => {
            eprintln!("Unknown operator: {}", args.op);
            eprintln!("Supported: rms_norm, silu, quantize_q8_k, vec_dot_q5_k, vec_dot_q6_k, vec_dot_q4_k");
            std::process::exit(1);
        }
    }
}
