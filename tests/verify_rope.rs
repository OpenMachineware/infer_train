// Verify ROPE correctness between Rust and llama.cpp

use infer_train::ops::rope;

mod ffi {
    unsafe extern "C" {
        pub fn llama_rope_neox_f32(
            n_dims: i32,
            n_heads: i32,
            dst: *mut f32,
            src: *const f32,
            pos: i32,
            freq_base: f32,
        );
    }
}

fn main() {
    let n_dims = 128;
    let n_heads = 32;
    let total = n_dims * n_heads;
    let pos = 0;
    let freq_base = 10000.0f32;

    // Generate test data
    let src: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut dst_llama = vec![0.0f32; total];
    let mut dst_rust = vec![0.0f32; total];

    unsafe {
        // llama.cpp
        ffi::llama_rope_neox_f32(
            n_dims as i32,
            n_heads as i32,
            dst_llama.as_mut_ptr(),
            src.as_ptr(),
            pos,
            freq_base,
        );

        // Rust
        for h in 0..n_heads {
            rope::rope_neox_simple(
                &src[h * n_dims..],
                &mut dst_rust[h * n_dims..],
                pos as usize,
                n_dims,
                freq_base,
            );
        }
    }

    // Compare results
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut total_diff = 0.0f32;

    for i in 0..total {
        let diff = (dst_llama[i] - dst_rust[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        total_diff += diff;
    }

    println!("ROPE Correctness Verification");
    println!("==============================");
    println!("n_dims: {}, n_heads: {}, total: {}", n_dims, n_heads, total);
    println!("pos: {}, freq_base: {}", pos, freq_base);
    println!();
    println!("Max diff: {:.6e} at index {}", max_diff, max_diff_idx);
    println!("Avg diff: {:.6e}", total_diff / total as f32);
    println!();

    // Print first few values
    println!("First 10 values:");
    println!("Index | llama.cpp | Rust      | Diff");
    println!("------|-----------|-----------|---------");
    for i in 0..10 {
        let diff = (dst_llama[i] - dst_rust[i]).abs();
        println!(
            "{:5} | {:9.4} | {:9.4} | {:.6e}",
            i, dst_llama[i], dst_rust[i], diff
        );
    }

    // Check a specific head
    println!("\nHead 0, first 5 pairs (before/after rotation):");
    for i in 0..5 {
        let idx1 = i;
        let idx2 = i + n_dims / 2;
        println!(
            "Pair {}: [{:.4}, {:.4}] vs [{:.4}, {:.4}]",
            i, dst_llama[idx1], dst_llama[idx2], dst_rust[idx1], dst_rust[idx2]
        );
    }

    if max_diff < 1e-5 {
        println!("\n✓ Results match (diff < 1e-5)");
    } else {
        println!("\n✗ Results differ significantly!");
    }
}
