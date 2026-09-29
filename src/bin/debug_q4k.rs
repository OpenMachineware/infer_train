// Q4_K插桩对比 - 检查Rust和llama.cpp的中间结果
// 使用完全相同的测试数据

use half::f16;
use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use std::ffi::c_void;

#[link(name = "ggml-cpu", kind = "dylib")]
unsafe extern "C" {
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
}

fn main() {
    unsafe {
        println!("Q4_K 插桩对比测试\n");

        // 使用完全相同的测试数据
        let k = 256; // 单个block
        let nb = 1;

        let mut x = BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales: [0u8; 12],
            qs: [0u8; 128],
        };

        // 填充特定值
        for j in 0..12 {
            x.scales[j] = (j + 1) as u8;
        }
        for j in 0..128 {
            x.qs[j] = (j as u8) & 0xFF;
        }

        let mut y = BlockQ8K { d: 1.0f32, qs: [0i8; 256], bsums: [0i16; 16] };

        for j in 0..256 {
            y.qs[j] = ((j % 64) - 32) as i8;
        }
        for j in 0..16 {
            y.bsums[j] = (j as i16) * 10;
        }

        // 调用llama.cpp
        let mut result_llama = 0.0f32;
        ggml_vec_dot_q4_K_q8_K(
            k as i32,
            &mut result_llama,
            0,
            &x as *const _ as *const c_void,
            0,
            &y as *const _ as *const c_void,
            0,
            1,
        );

        // 调用Rust实现
        let result_rust =
            infer_train::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon(
                k,
                std::slice::from_ref(&x),
                std::slice::from_ref(&y),
            );

        println!("结果对比:");
        println!("  llama.cpp: {}", result_llama);
        println!("  Rust:      {}", result_rust);
        println!("  差异:      {}", (result_llama - result_rust).abs());

        if (result_llama - result_rust).abs() < 1e-5 {
            println!("  ✓ 结果一致");
        } else {
            println!("  ✗ 结果不一致！");
        }

        // 性能对比
        println!("\n性能对比 (K=256, 100000次迭代):");

        let iterations = 100000;

        // llama.cpp
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            ggml_vec_dot_q4_K_q8_K(
                k as i32,
                &mut result_llama,
                0,
                &x as *const _ as *const c_void,
                0,
                &y as *const _ as *const c_void,
                0,
                1,
            );
        }
        let elapsed_llama = start.elapsed();

        // Rust
        let x_slice = std::slice::from_ref(&x);
        let y_slice = std::slice::from_ref(&y);
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            result_rust =
                infer_train::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon(
                    k, x_slice, y_slice,
                );
        }
        let elapsed_rust = start.elapsed();

        let flops = 2.0 * (k as f64) * (iterations as f64);
        let gflops_llama = flops / elapsed_llama.as_secs_f64() / 1e9;
        let gflops_rust = flops / elapsed_rust.as_secs_f64() / 1e9;

        println!(
            "  llama.cpp: {:.1} GFLOPS ({:.2} ms)",
            gflops_llama,
            elapsed_llama.as_secs_f64() * 1000.0
        );
        println!(
            "  Rust:      {:.1} GFLOPS ({:.2} ms)",
            gflops_rust,
            elapsed_rust.as_secs_f64() * 1000.0
        );
        println!("  Ratio:     {:.1}%", (gflops_rust / gflops_llama) * 100.0);
    }
}
