// CPU operators benchmark: llama.cpp vs Rust
// Tests all operators with direct comparison
// Method: 10000 iterations, no warmup, release mode

use std::hint::black_box;
use std::time::Instant;

// FFI bindings to llama.cpp operators
mod ffi {
    unsafe extern "C" {
        // Activation functions
        pub fn llama_silu_f32(n: i32, y: *mut f32, x: *const f32);
        pub fn llama_gelu_f32(n: i32, y: *mut f32, x: *const f32);
        pub fn llama_gelu_quick_f32(n: i32, y: *mut f32, x: *const f32);
        pub fn llama_softmax_f32(n: i32, y: *mut f32, x: *const f32) -> f32;

        // Binary operations
        pub fn llama_add_f32(n: i32, z: *mut f32, x: *const f32, y: *const f32);
        pub fn llama_sub_f32(n: i32, z: *mut f32, x: *const f32, y: *const f32);
        pub fn llama_mul_f32(n: i32, z: *mut f32, x: *const f32, y: *const f32);
        pub fn llama_div_f32(n: i32, z: *mut f32, x: *const f32, y: *const f32);

        // RMS_NORM, ROPE, SCALE
        pub fn llama_rms_norm_f32(
            n_rows: i32,
            hidden_dim: i32,
            dst: *mut f32,
            x: *const f32,
            w: *const f32,
            eps: f32,
        );
        pub fn llama_rope_neox_f32(
            n_dims: i32,
            n_heads: i32,
            dst: *mut f32,
            src: *const f32,
            pos: i32,
            freq_base: f32,
        );
        pub fn llama_scale_f32(
            n: i32,
            dst: *mut f32,
            src: *const f32,
            scale: f32,
        );
        pub fn llama_add1_f32(
            n: i32,
            dst: *mut f32,
            src: *const f32,
            value: f32,
        );
        pub fn llama_dup_f32(n: i32, dst: *mut f32, src: *const f32);
    }
}

const ITERATIONS: usize = 10000;

fn generate_test_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * 0.1 - n as f32 * 0.05).sin()).collect()
}

fn generate_div_safe_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| 1.0 + (i as f32 * 0.01).abs()).collect()
}

fn print_result(name: &str, llama_ns: f64, rust_ns: f64, n: usize) {
    let llama_per_elem = llama_ns / n as f64;
    let rust_per_elem = rust_ns / n as f64;
    let ratio = llama_ns / rust_ns;
    let status = if ratio >= 1.0 { "✓" } else { "✗" };
    println!(
        "{:<15} | {:6.2} | {:6.2} | {:5.0}% | {}",
        name,
        llama_per_elem,
        rust_per_elem,
        ratio * 100.0,
        status
    );
}

fn main() {
    println!("=== CPU Operators Performance Comparison ===");
    println!("Method: {} iterations, no warmup, same process\n", ITERATIONS);

    println!("Operator        | llama.cpp | Rust   | Ratio | Status");
    println!("----------------|-----------|--------|-------|-------");

    unsafe {
        use ffi::*;
        use infer_train::ops::{
            binary, dup, gelu, rms_norm, rope, scalar, silu, softmax,
        };

        let n = 4096;
        let x = generate_test_data(n);
        let mut y_llama = vec![0.0f32; n];
        let mut y_rust = vec![0.0f32; n];

        // SILU
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_silu_f32(n as i32, y_llama.as_mut_ptr(), x.as_ptr());
            black_box(y_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            silu::silu_f32(&x, &mut y_rust);
            black_box(y_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("SILU", llama_ns, rust_ns, n);

        // GELU
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_gelu_f32(n as i32, y_llama.as_mut_ptr(), x.as_ptr());
            black_box(y_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            gelu::gelu_f32(&x, &mut y_rust);
            black_box(y_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("GELU", llama_ns, rust_ns, n);

        // GELU_QUICK
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_gelu_quick_f32(n as i32, y_llama.as_mut_ptr(), x.as_ptr());
            black_box(y_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            gelu::gelu_quick_f32(&x, &mut y_rust);
            black_box(y_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("GELU_QUICK", llama_ns, rust_ns, n);

        // SOFTMAX
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_softmax_f32(n as i32, y_llama.as_mut_ptr(), x.as_ptr());
            black_box(y_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            softmax::softmax_f32_neon(&x, &mut y_rust);
            black_box(y_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("SOFTMAX", llama_ns, rust_ns, n);

        println!("\n--- Binary Operations ---");

        let y = generate_test_data(n);

        // ADD
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_add_f32(
                n as i32,
                y_llama.as_mut_ptr(),
                x.as_ptr(),
                y.as_ptr(),
            );
            black_box(y_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            binary::add_f32(&x, &y, &mut y_rust);
            black_box(y_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("ADD", llama_ns, rust_ns, n);

        // SUB
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_sub_f32(
                n as i32,
                y_llama.as_mut_ptr(),
                x.as_ptr(),
                y.as_ptr(),
            );
            black_box(y_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            binary::sub_f32(&x, &y, &mut y_rust);
            black_box(y_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("SUB", llama_ns, rust_ns, n);

        // MUL
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_mul_f32(
                n as i32,
                y_llama.as_mut_ptr(),
                x.as_ptr(),
                y.as_ptr(),
            );
            black_box(y_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            binary::mul_f32(&x, &y, &mut y_rust);
            black_box(y_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("MUL", llama_ns, rust_ns, n);

        // DIV
        let y_safe = generate_div_safe_data(n);
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_div_f32(
                n as i32,
                y_llama.as_mut_ptr(),
                x.as_ptr(),
                y_safe.as_ptr(),
            );
            black_box(y_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            binary::div_f32(&x, &y_safe, &mut y_rust);
            black_box(y_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("DIV", llama_ns, rust_ns, n);

        println!("\n--- Other Operations ---");

        // RMS_NORM
        let hidden_dim = 4096;
        let n_rows = 1;
        let x_rms = generate_test_data(hidden_dim);
        let w: Vec<f32> =
            (0..hidden_dim).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut dst_llama = vec![0.0f32; hidden_dim];
        let mut dst_rust = vec![0.0f32; hidden_dim];
        let eps = 1e-5f32;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_rms_norm_f32(
                n_rows as i32,
                hidden_dim as i32,
                dst_llama.as_mut_ptr(),
                x_rms.as_ptr(),
                w.as_ptr(),
                eps,
            );
            black_box(dst_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            rms_norm::rms_norm_f32(&x_rms, &w, &mut dst_rust, hidden_dim, eps);
            black_box(dst_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("RMS_NORM", llama_ns, rust_ns, hidden_dim);

        // ROPE
        let n_dims = 128;
        let n_heads = 32;
        let total = n_dims * n_heads;
        let src = generate_test_data(total);
        let mut dst_llama = vec![0.0f32; total];
        let mut dst_rust = vec![0.0f32; total];
        let pos = 0;
        let freq_base = 10000.0f32;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_rope_neox_f32(
                n_dims as i32,
                n_heads as i32,
                dst_llama.as_mut_ptr(),
                src.as_ptr(),
                pos,
                freq_base,
            );
            black_box(dst_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            for h in 0..n_heads {
                rope::rope_neox_simple(
                    &src[h * n_dims..],
                    &mut dst_rust[h * n_dims..],
                    pos as usize,
                    n_dims,
                    freq_base,
                );
            }
            black_box(dst_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("ROPE", llama_ns, rust_ns, total);

        // SCALE
        let scale = 0.5f32;
        let mut dst_llama = vec![0.0f32; n];
        let mut dst_rust = vec![0.0f32; n];

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_scale_f32(
                n as i32,
                dst_llama.as_mut_ptr(),
                x.as_ptr(),
                scale,
            );
            black_box(dst_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            scalar::scale_f32(&x, &mut dst_rust, scale);
            black_box(dst_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("SCALE", llama_ns, rust_ns, n);

        // ADD1: dst = src + value
        let value = 1.0f32;
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_add1_f32(n as i32, dst_llama.as_mut_ptr(), x.as_ptr(), value);
            black_box(dst_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            scalar::add1_f32(&x, &mut dst_rust, value);
            black_box(dst_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("ADD1", llama_ns, rust_ns, n);

        // DUP/CPY: memcpy
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            llama_dup_f32(n as i32, dst_llama.as_mut_ptr(), x.as_ptr());
            black_box(dst_llama[0]);
        }
        let llama_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            dup::dup_same(&x, &mut dst_rust);
            black_box(dst_rust[0]);
        }
        let rust_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        print_result("DUP", llama_ns, rust_ns, n);
    }

    println!("\n✓ = Rust faster than llama.cpp");
    println!("✗ = Rust slower than llama.cpp");
    println!("\nNote: Times are ns/element");
}
