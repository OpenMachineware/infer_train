// Verify RMSNorm correctness against llama.cpp
// Run: cargo run --release --bin verify_rms_norm

use infer_train::ops::rms_norm::rms_norm_f32;

fn main() {
    println!("=== RMSNorm Correctness Verification ===\n");

    let hidden_dim = 4096;
    let n_rows = 1;
    let eps = 1e-5;

    // Create test data
    let x: Vec<f32> = (0..n_rows * hidden_dim)
        .map(|i| ((i % 100) as f32 * 0.01 - 0.5))
        .collect();

    let w: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();

    let mut dst = vec![0.0f32; n_rows * hidden_dim];

    // Run our implementation
    unsafe {
        rms_norm_f32(&x, &w, &mut dst, hidden_dim, eps);
    }

    // Compute expected result manually
    let mut expected = vec![0.0f32; hidden_dim];
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();

    for i in 0..hidden_dim {
        expected[i] = x[i] * scale * w[i];
    }

    // Compare
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;
    for i in 0..hidden_dim {
        let error = (dst[i] - expected[i]).abs();
        max_error = max_error.max(error);
        total_error += error;
    }
    let avg_error = total_error / hidden_dim as f32;

    println!("Hidden dim: {}", hidden_dim);
    println!("Max error: {}", max_error);
    println!("Avg error: {}", avg_error);
    println!();

    // Show first 5 elements
    println!("First 5 elements:");
    for i in 0..5 {
        println!(
            "  x={:.6}, w={:.6}, expected={:.6}, got={:.6}, error={:.9}",
            x[i],
            w[i],
            expected[i],
            dst[i],
            (dst[i] - expected[i]).abs()
        );
    }

    // Check if within tolerance
    let tolerance = 1e-6;
    if max_error < tolerance {
        println!("\n✓ PASS: Max error {} < tolerance {}", max_error, tolerance);
    } else {
        println!("\n✗ FAIL: Max error {} >= tolerance {}", max_error, tolerance);
    }

    // Test multiple rows
    println!("\n=== Multi-row Test ===");
    let n_rows = 4;
    let x_multi: Vec<f32> = (0..n_rows * hidden_dim)
        .map(|i| ((i % 100) as f32 * 0.01 - 0.5))
        .collect();

    let mut dst_multi = vec![0.0f32; n_rows * hidden_dim];

    unsafe {
        rms_norm_f32(&x_multi, &w, &mut dst_multi, hidden_dim, eps);
    }

    // Verify each row
    for row in 0..n_rows {
        let offset = row * hidden_dim;
        let x_row = &x_multi[offset..offset + hidden_dim];
        let dst_row = &dst_multi[offset..offset + hidden_dim];

        let sum_sq: f32 = x_row.iter().map(|&v| v * v).sum();
        let mean_sq = sum_sq / hidden_dim as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();

        let mut row_max_error = 0.0f32;
        for i in 0..hidden_dim {
            let expected = x_row[i] * scale * w[i];
            let error = (dst_row[i] - expected).abs();
            row_max_error = row_max_error.max(error);
        }

        println!("Row {}: max_error = {:.9}", row, row_max_error);
    }
}
