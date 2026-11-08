#!/usr/bin/env cargo
//! 测量GEMM性能，找出差距

use std::time::{Duration, Instant};

fn main() {
    println!("=== GEMM性能对比 ===\n");

    // 测试配置
    let m = 64;
    let k = 896;
    let n = 896;

    println!("矩阵尺寸: {}×{}×{}\n", m, k, n);

    // 创建测试数据
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).cos()).collect();
    let mut c: Vec<f32> = vec![0.0; m * n];

    let iterations = 100;

    // 测试我们的simd_gemm
    let mut total_time = Duration::new(0, 0);

    for _ in 0..iterations {
        let start = Instant::now();
        unsafe {
            infer_train::cpu::simd_gemm::simd_gemm(
                c.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                m, k, n,
            );
        }
        total_time += start.elapsed();
        std::hint::black_box(&c);
    }

    let avg_time = total_time / iterations;
    let flops = 2 * m * k * n;
    let gflops = flops as f64 * iterations as f64 / total_time.as_secs_f64() / 1e9;

    println!("我们的simd_gemm:");
    println!("  平均时间: {:?}", avg_time);
    println!("  GFLOPS: {:.1}", gflops);

    // 理论峰值
    // M1 CPU: 4性能核心 × 4 SIMD宽度 × 2 (FMA) × 3.2 GHz = 102.4 GFLOPS per core
    // 如果用8线程: 理论 ~200 GFLOPS
    let theoretical_peak = 200.0;
    println!("  理论峰值: {:.0} GFLOPS", theoretical_peak);
    println!("  效率: {:.1}%", gflops / theoretical_peak * 100.0);

    // 内存带宽分析
    // 输入A: m×k = 64×896 = 57KB
    // 输入B: k×n = 896×896 = 3.2MB
    // 输出C: m×n = 64×896 = 224KB
    let data_read = (m * k + k * n) * 4;
    let data_write = m * n * 4;
    let total_data = data_read + data_write;
    let bandwidth = total_data as f64 / avg_time.as_secs_f64() / 1e9;

    println!("\n内存分析:");
    println!("  数据读取: {:.1} KB", data_read as f64 / 1024.0);
    println!("  数据写入: {:.1} KB", data_write as f64 / 1024.0);
    println!("  总数据量: {:.1} KB", total_data as f64 / 1024.0);
    println!("  带宽: {:.1} GB/s", bandwidth);

    // 计算强度
    let arithmetic_intensity = flops as f64 / total_data as f64;
    println!("  计算强度: {:.1} FLOP/Byte", arithmetic_intensity);

    // 理论时间（基于内存带宽）
    let m1_bandwidth = 100.0; // GB/s (L2 cache)
    let memory_bound_time = total_data as f64 / m1_bandwidth / 1e9 * 1e6;
    println!("  内存bound理论时间: {:.1} µs", memory_bound_time);
    println!("  实际时间: {:.1} µs", avg_time.as_micros() as f64);
}
