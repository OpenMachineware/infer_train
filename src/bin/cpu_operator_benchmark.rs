// 统一CPU算子性能对比测试框架
// 每个算子、每种格式、每个size，直接与llama.cpp对比

use std::ffi::c_void;
use std::hint::black_box;
use std::time::{Duration, Instant};

// ===== 测试配置 =====
const SIZES: &[usize] = &[256, 512, 1024, 2048, 4096];
const ITERATIONS_SMALL: usize = 50000; // K <= 1024
const ITERATIONS_LARGE: usize = 10000; // K > 1024
const WARMUP_ITERATIONS: usize = 100_000_000;
const WARMUP_SLEEP_MS: u64 = 1000;

// ===== llama.cpp FFI 声明 =====
#[link(name = "ggml-cpu", kind = "dylib")]
unsafe extern "C" {
    // K-quant
    fn ggml_vec_dot_q2_K_q8_K(
        n: i32,
        s: *mut f32,
        bs: usize,
        vx: *const c_void,
        bx: usize,
        vy: *const c_void,
        by: usize,
        nrc: i32,
    );
    fn ggml_vec_dot_q3_K_q8_K(
        n: i32,
        s: *mut f32,
        bs: usize,
        vx: *const c_void,
        bx: usize,
        vy: *const c_void,
        by: usize,
        nrc: i32,
    );
    fn ggml_vec_dot_q4_K_q8_K(
        n: i32,
        s: *mut f32,
        bs: usize,
        vx: *const c_void,
        bx: usize,
        vy: *const c_void,
        by: usize,
        nrc: i32,
    );
    fn ggml_vec_dot_q5_K_q8_K(
        n: i32,
        s: *mut f32,
        bs: usize,
        vx: *const c_void,
        bx: usize,
        vy: *const c_void,
        by: usize,
        nrc: i32,
    );
    fn ggml_vec_dot_q6_K_q8_K(
        n: i32,
        s: *mut f32,
        bs: usize,
        vx: *const c_void,
        bx: usize,
        vy: *const c_void,
        by: usize,
        nrc: i32,
    );
}

// ===== 测试结果 =====
struct BenchmarkResult {
    format: &'static str,
    size: usize,
    rust_gflops: f64,
    llama_gflops: f64,
    ratio: f64,
}

// ===== CPU预热 =====
fn warmup_cpu() {
    let mut sum = 0.0f64;
    for i in 0..WARMUP_ITERATIONS {
        sum = sum.sqrt() + i as f64;
    }
    std::thread::sleep(Duration::from_millis(WARMUP_SLEEP_MS));
    let _ = black_box(sum);
}

// ===== 辅助函数 =====
fn calc_gflops(k: usize, iterations: usize, elapsed: Duration) -> f64 {
    let flops = 2.0 * (k as f64) * (iterations as f64);
    flops / elapsed.as_secs_f64() / 1e9
}

// ===== K-quant vec_dot 测试 =====
#[cfg(target_arch = "aarch64")]
mod k_quant_tests {
    use super::*;
    use half::f16;
    use infer_train::quant::types::{
        BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K,
    };
    use infer_train::quant::vec_dot::arm::{
        vec_dot_q2_k_q8_k_neon, vec_dot_q3_k_q8_k_neon, vec_dot_q4_k_q8_k_neon,
        vec_dot_q5_k_q8_k_neon, vec_dot_q6_k_q8_k_neon,
    };

    // 生成统一的测试数据
    fn generate_q8k_blocks(nb: usize, seed: usize) -> Vec<BlockQ8K> {
        (0..nb)
            .map(|i| {
                let mut qs = [0i8; 256];
                let mut bsums = [0i16; 16];
                for j in 0..256 {
                    qs[j] = (((i * 17 + j * 13 + seed) % 128) - 64) as i8;
                }
                for j in 0..16 {
                    bsums[j] = ((i + j + seed) % 256) as i16;
                }
                BlockQ8K { d: 1.0 + (i % 10) as f32 * 0.1, qs, bsums }
            })
            .collect()
    }

    fn generate_q2k_blocks(nb: usize, seed: usize) -> Vec<BlockQ2K> {
        (0..nb)
            .map(|i| {
                let mut qs = [0u8; 64];
                let mut scales = [0u8; 16];
                for j in 0..64 {
                    qs[j] = ((i + j + seed) & 0xFF) as u8;
                }
                for j in 0..16 {
                    scales[j] = ((j % 4) + 1) as u8;
                }
                BlockQ2K {
                    d: f16::from_f32(0.5).to_bits(),
                    dmin: f16::from_f32(0.0).to_bits(),
                    scales,
                    qs,
                }
            })
            .collect()
    }

    fn generate_q3k_blocks(nb: usize, seed: usize) -> Vec<BlockQ3K> {
        (0..nb)
            .map(|i| {
                let mut hmask = [0u8; 32];
                let mut qs = [0u8; 64];
                let mut scales = [0u8; 12];
                for j in 0..32 {
                    hmask[j] = ((i * j + seed) & 0xFF) as u8;
                }
                for j in 0..64 {
                    qs[j] = ((i + j + seed) & 0xFF) as u8;
                }
                for j in 0..12 {
                    scales[j] = ((j % 4) + 1) as u8;
                }
                BlockQ3K { d: f16::from_f32(1.0).to_bits(), hmask, qs, scales }
            })
            .collect()
    }

    fn generate_q4k_blocks(nb: usize, seed: usize) -> Vec<BlockQ4K> {
        (0..nb)
            .map(|i| {
                let mut scales = [0u8; 12];
                let mut qs = [0u8; 128];
                for j in 0..12 {
                    scales[j] = ((j % 4) + 1) as u8;
                }
                for j in 0..128 {
                    qs[j] = ((i + j + seed) & 0xFF) as u8;
                }
                BlockQ4K {
                    d: f16::from_f32(1.0).to_bits(),
                    dmin: f16::from_f32(0.0).to_bits(),
                    scales,
                    qs,
                }
            })
            .collect()
    }

    fn generate_q5k_blocks(nb: usize, seed: usize) -> Vec<BlockQ5K> {
        (0..nb)
            .map(|i| {
                let mut qs = [0u8; 128];
                let mut qh = [0u8; 32];
                let mut scales = [0u8; 12];
                for j in 0..128 {
                    qs[j] = ((i + j + seed) & 0xFF) as u8;
                }
                for j in 0..32 {
                    qh[j] = ((i * j + seed) & 0xFF) as u8;
                }
                for j in 0..12 {
                    scales[j] = ((j % 4) + 1) as u8;
                }
                BlockQ5K {
                    d: f16::from_f32(1.0).to_bits(),
                    dmin: f16::from_f32(0.0).to_bits(),
                    qs,
                    qh,
                    scales,
                }
            })
            .collect()
    }

    fn generate_q6k_blocks(nb: usize, seed: usize) -> Vec<BlockQ6K> {
        (0..nb)
            .map(|i| {
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                let mut scales = [0i8; 16];
                for j in 0..128 {
                    ql[j] = ((i + j + seed) & 0xFF) as u8;
                }
                for j in 0..64 {
                    qh[j] = ((i * j + seed) & 0xFF) as u8;
                }
                for j in 0..16 {
                    scales[j] = ((j % 4) + 1) as i8;
                }
                BlockQ6K { d: f16::from_f32(1.0).to_bits(), ql, qh, scales }
            })
            .collect()
    }

    pub fn benchmark_k_quant() -> Vec<BenchmarkResult> {
        unsafe {
            let mut results = Vec::new();
            let seed = 42;

            println!(
                "\n=== K-quant vec_dot 性能对比 (Rust vs llama.cpp) ===\n"
            );
            println!(
                "Format | K     | Rust GFLOPS | llama.cpp | Ratio | Status"
            );
            println!(
                "-------|-------|-------------|-----------|-------|--------"
            );

            for &k in SIZES {
                let nb = k / QK_K;
                let iterations =
                    if k <= 1024 { ITERATIONS_SMALL } else { ITERATIONS_LARGE };
                let y = generate_q8k_blocks(nb, seed);

                // Q2_K
                {
                    let x = generate_q2k_blocks(nb, seed);
                    let mut result = 0.0f32;

                    let start = Instant::now();
                    for _ in 0..iterations {
                        result = black_box(vec_dot_q2_k_q8_k_neon(k, &x, &y));
                    }
                    let elapsed_rust = start.elapsed();
                    let rust_gflops = calc_gflops(k, iterations, elapsed_rust);

                    let start = Instant::now();
                    for _ in 0..iterations {
                        ggml_vec_dot_q2_K_q8_K(
                            k as i32,
                            &mut result,
                            0,
                            x.as_ptr() as *const c_void,
                            0,
                            y.as_ptr() as *const c_void,
                            0,
                            1,
                        );
                    }
                    let elapsed_llama = start.elapsed();
                    let llama_gflops =
                        calc_gflops(k, iterations, elapsed_llama);

                    let ratio = (rust_gflops / llama_gflops) * 100.0;
                    let status = if ratio >= 100.0 { "✓" } else { "✗" };

                    println!(
                        "Q2_K   | {:5} | {:11.1} | {:9.1} | {:5.0}% | {}",
                        k, rust_gflops, llama_gflops, ratio, status
                    );

                    results.push(BenchmarkResult {
                        format: "Q2_K",
                        size: k,
                        rust_gflops,
                        llama_gflops,
                        ratio,
                    });
                }

                // Q3_K
                {
                    let x = generate_q3k_blocks(nb, seed);
                    let mut result = 0.0f32;

                    let start = Instant::now();
                    for _ in 0..iterations {
                        result = black_box(vec_dot_q3_k_q8_k_neon(k, &x, &y));
                    }
                    let elapsed_rust = start.elapsed();
                    let rust_gflops = calc_gflops(k, iterations, elapsed_rust);

                    let start = Instant::now();
                    for _ in 0..iterations {
                        ggml_vec_dot_q3_K_q8_K(
                            k as i32,
                            &mut result,
                            0,
                            x.as_ptr() as *const c_void,
                            0,
                            y.as_ptr() as *const c_void,
                            0,
                            1,
                        );
                    }
                    let elapsed_llama = start.elapsed();
                    let llama_gflops =
                        calc_gflops(k, iterations, elapsed_llama);

                    let ratio = (rust_gflops / llama_gflops) * 100.0;
                    let status = if ratio >= 100.0 { "✓" } else { "✗" };

                    println!(
                        "Q3_K   | {:5} | {:11.1} | {:9.1} | {:5.0}% | {}",
                        k, rust_gflops, llama_gflops, ratio, status
                    );

                    results.push(BenchmarkResult {
                        format: "Q3_K",
                        size: k,
                        rust_gflops,
                        llama_gflops,
                        ratio,
                    });
                }

                // Q4_K
                {
                    let x = generate_q4k_blocks(nb, seed);
                    let mut result = 0.0f32;

                    let start = Instant::now();
                    for _ in 0..iterations {
                        result = black_box(vec_dot_q4_k_q8_k_neon(k, &x, &y));
                    }
                    let elapsed_rust = start.elapsed();
                    let rust_gflops = calc_gflops(k, iterations, elapsed_rust);

                    let start = Instant::now();
                    for _ in 0..iterations {
                        ggml_vec_dot_q4_K_q8_K(
                            k as i32,
                            &mut result,
                            0,
                            x.as_ptr() as *const c_void,
                            0,
                            y.as_ptr() as *const c_void,
                            0,
                            1,
                        );
                    }
                    let elapsed_llama = start.elapsed();
                    let llama_gflops =
                        calc_gflops(k, iterations, elapsed_llama);

                    let ratio = (rust_gflops / llama_gflops) * 100.0;
                    let status = if ratio >= 100.0 { "✓" } else { "✗" };

                    println!(
                        "Q4_K   | {:5} | {:11.1} | {:9.1} | {:5.0}% | {}",
                        k, rust_gflops, llama_gflops, ratio, status
                    );

                    results.push(BenchmarkResult {
                        format: "Q4_K",
                        size: k,
                        rust_gflops,
                        llama_gflops,
                        ratio,
                    });
                }

                // Q5_K
                {
                    let x = generate_q5k_blocks(nb, seed);
                    let mut result = 0.0f32;

                    let start = Instant::now();
                    for _ in 0..iterations {
                        result = black_box(vec_dot_q5_k_q8_k_neon(k, &x, &y));
                    }
                    let elapsed_rust = start.elapsed();
                    let rust_gflops = calc_gflops(k, iterations, elapsed_rust);

                    let start = Instant::now();
                    for _ in 0..iterations {
                        ggml_vec_dot_q5_K_q8_K(
                            k as i32,
                            &mut result,
                            0,
                            x.as_ptr() as *const c_void,
                            0,
                            y.as_ptr() as *const c_void,
                            0,
                            1,
                        );
                    }
                    let elapsed_llama = start.elapsed();
                    let llama_gflops =
                        calc_gflops(k, iterations, elapsed_llama);

                    let ratio = (rust_gflops / llama_gflops) * 100.0;
                    let status = if ratio >= 100.0 { "✓" } else { "✗" };

                    println!(
                        "Q5_K   | {:5} | {:11.1} | {:9.1} | {:5.0}% | {}",
                        k, rust_gflops, llama_gflops, ratio, status
                    );

                    results.push(BenchmarkResult {
                        format: "Q5_K",
                        size: k,
                        rust_gflops,
                        llama_gflops,
                        ratio,
                    });
                }

                // Q6_K
                {
                    let x = generate_q6k_blocks(nb, seed);
                    let mut result = 0.0f32;

                    let start = Instant::now();
                    for _ in 0..iterations {
                        result = black_box(vec_dot_q6_k_q8_k_neon(k, &x, &y));
                    }
                    let elapsed_rust = start.elapsed();
                    let rust_gflops = calc_gflops(k, iterations, elapsed_rust);

                    let start = Instant::now();
                    for _ in 0..iterations {
                        ggml_vec_dot_q6_K_q8_K(
                            k as i32,
                            &mut result,
                            0,
                            x.as_ptr() as *const c_void,
                            0,
                            y.as_ptr() as *const c_void,
                            0,
                            1,
                        );
                    }
                    let elapsed_llama = start.elapsed();
                    let llama_gflops =
                        calc_gflops(k, iterations, elapsed_llama);

                    let ratio = (rust_gflops / llama_gflops) * 100.0;
                    let status = if ratio >= 100.0 { "✓" } else { "✗" };

                    println!(
                        "Q6_K   | {:5} | {:11.1} | {:9.1} | {:5.0}% | {}",
                        k, rust_gflops, llama_gflops, ratio, status
                    );

                    results.push(BenchmarkResult {
                        format: "Q6_K",
                        size: k,
                        rust_gflops,
                        llama_gflops,
                        ratio,
                    });
                }

                println!();
            }

            results
        }
    }
}

// ===== 主函数 =====
fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     CPU算子统一性能对比测试（vs llama.cpp）                 ║");
    println!("║     每个算子 · 每种格式 · 每个尺寸                          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("\n使用: llama.cpp-0.6.0/build/bin/libggml-cpu.dylib");
    println!("数据: 统一种子 (seed=42)，可复现\n");

    println!("预热CPU...");
    warmup_cpu();
    println!("预热完成");

    #[cfg(target_arch = "aarch64")]
    {
        let all_results = k_quant_tests::benchmark_k_quant();

        // 汇总报告
        println!(
            "╔════════════════════════════════════════════════════════════╗"
        );
        println!(
            "║                     汇总报告                                ║"
        );
        println!(
            "╚════════════════════════════════════════════════════════════╝\n"
        );

        let total = all_results.len();
        let passed = all_results.iter().filter(|r| r.ratio >= 100.0).count();

        println!("总计: {} 项测试", total);
        println!("通过: {} 项 (≥100%)", passed);
        println!("未过: {} 项 (<100%)", total - passed);

        if passed == total {
            println!("\n✅ 所有算子性能均超过llama.cpp！");
        } else {
            println!("\n⚠️  部分算子需要优化:");
            for r in &all_results {
                if r.ratio < 100.0 {
                    println!(
                        "  - {} @ K={}: {:.0}%",
                        r.format, r.size, r.ratio
                    );
                }
            }
        }

        // 保存结果
        println!("\n保存结果到 docs/cpu_operator_performance.md");
        let mut md = String::new();
        md.push_str("# CPU算子性能对比测试结果\n\n");
        md.push_str("## 测试环境\n");
        md.push_str("- 平台: Apple M1 (ARM64)\n");
        md.push_str("- 对比: Rust NEON vs llama.cpp (直接调用dylib)\n");
        md.push_str("- 数据: 统一种子 (seed=42)\n\n");
        md.push_str("## K-quant vec_dot 性能\n\n");
        md.push_str(
            "| Format | K | Rust (GFLOPS) | llama.cpp (GFLOPS) | Ratio |\n",
        );
        md.push_str(
            "|--------|---|---------------|-------------------|-------|\n",
        );
        for r in &all_results {
            md.push_str(&format!(
                "| {} | {} | {:.1} | {:.1} | {:.0}% |\n",
                r.format, r.size, r.rust_gflops, r.llama_gflops, r.ratio
            ));
        }
        md.push_str(&format!("\n**汇总**: {}/{} 项通过\n", passed, total));
        std::fs::write("docs/cpu_operator_performance.md", md).ok();
    }
}
