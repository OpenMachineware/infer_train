// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

/// CPU reference implementation for SSM Conv1D
/// Input x: [d_inner, n_tok + d_conv - 1]
/// Weight w: [d_inner, d_conv]
/// Output y: [d_inner, n_tok]
fn ssm_conv1d_cpu(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    d_inner: usize,
    d_conv: usize,
    n_tok: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; d_inner * n_tok];

    for d in 0..d_inner {
        for t in 0..n_tok {
            let mut sum = 0.0f32;
            for j in 0..d_conv {
                let x_idx = d * (n_tok + d_conv - 1) + t + j;
                let w_idx = d * d_conv + j;
                sum += x[x_idx] * w[w_idx];
            }
            if let Some(b) = bias {
                sum += b[d];
            }
            y[d * n_tok + t] = sum;
        }
    }

    y
}

/// CPU reference implementation for SSM Scan (Mamba-1 style)
/// Input x: [d_inner, n_tok]
/// dt: [d_inner, n_tok]
/// A: [d_inner, d_state]
/// B: [d_state, n_tok]
/// C: [d_state, n_tok]
/// s0: [d_inner, d_state] (initial state, optional)
/// Output y: [d_inner, n_tok], s: [d_inner, d_state]
fn ssm_scan_cpu(
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    s0: Option<&[f32]>,
    d_inner: usize,
    d_state: usize,
    n_tok: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; d_inner * n_tok];
    let mut s = vec![0.0f32; d_inner * d_state];

    // Initialize state
    for d in 0..d_inner {
        for n in 0..d_state {
            s[d * d_state + n] =
                s0.map(|s0| s0[d * d_state + n]).unwrap_or(0.0);
        }
    }

    // Scan over time
    for d in 0..d_inner {
        for t in 0..n_tok {
            let x_t = x[d * n_tok + t];
            let dt_t = dt[d * n_tok + t];

            // Softplus on dt
            let dt_sp =
                if dt_t <= 20.0 { (1.0 + dt_t.exp()).ln() } else { dt_t };

            // x * dt
            let x_dt = x_t * dt_sp;

            // Accumulate output
            let mut y_t = 0.0f32;

            // Update state and compute output
            for n in 0..d_state {
                let a_val = a[d * d_state + n];
                let b_t = b[n * n_tok + t];
                let c_t = c[n * n_tok + t];

                // Decay
                let decay = (dt_sp * a_val).exp();

                // State update
                s[d * d_state + n] = s[d * d_state + n] * decay + b_t * x_dt;

                // Output
                y_t += s[d * d_state + n] * c_t;
            }

            y[d * n_tok + t] = y_t;
        }
    }

    (y, s)
}

#[cfg(target_arch = "aarch64")]
mod gpu_tests {
    use super::*;

    #[test]
    fn test_ssm_conv1d_cpu() {
        // Test parameters (small scale for debugging)
        let d_inner = 4;
        let d_conv = 4;
        let n_tok = 8;

        // Generate test data
        let mut x = Vec::with_capacity(d_inner * (n_tok + d_conv - 1));
        let mut w = Vec::with_capacity(d_inner * d_conv);
        let mut bias = Vec::with_capacity(d_inner);

        for i in 0..d_inner * (n_tok + d_conv - 1) {
            x.push((i as f32 * 0.1) - 1.0);
        }
        for i in 0..d_inner * d_conv {
            w.push((i as f32 * 0.05) - 0.1);
        }
        for i in 0..d_inner {
            bias.push(i as f32 * 0.01);
        }

        // CPU reference
        let y_cpu = ssm_conv1d_cpu(&x, &w, Some(&bias), d_inner, d_conv, n_tok);

        // Verify output size
        assert_eq!(y_cpu.len(), d_inner * n_tok);

        // Print first few outputs
        println!("SSM Conv1D CPU output (first 8):");
        for i in 0..8.min(y_cpu.len()) {
            println!("  y[{}] = {}", i, y_cpu[i]);
        }

        // Verify no NaN
        for (i, &val) in y_cpu.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at y[{}]", i);
            assert!(!val.is_infinite(), "Inf at y[{}]", i);
        }
    }

    #[test]
    fn test_ssm_scan_cpu() {
        // Test parameters
        let d_inner = 4;
        let d_state = 16;
        let n_tok = 8;

        // Generate test data
        let mut x = Vec::with_capacity(d_inner * n_tok);
        let mut dt = Vec::with_capacity(d_inner * n_tok);
        let mut a = Vec::with_capacity(d_inner * d_state);
        let mut b = Vec::with_capacity(d_state * n_tok);
        let mut c = Vec::with_capacity(d_state * n_tok);

        // Use small values to avoid overflow
        for i in 0..d_inner * n_tok {
            x.push((i as f32 * 0.01) - 0.1);
            dt.push((i as f32 * 0.001) - 0.01); // Small dt for stability
        }
        for i in 0..d_inner * d_state {
            // A is typically negative (decay)
            a.push(-((i % 10) as f32 + 1.0) * 0.1);
        }
        for i in 0..d_state * n_tok {
            b.push((i as f32 * 0.01) - 0.1);
            c.push((i as f32 * 0.01) - 0.1);
        }

        // CPU reference
        let (y_cpu, s_cpu) =
            ssm_scan_cpu(&x, &dt, &a, &b, &c, None, d_inner, d_state, n_tok);

        // Verify output size
        assert_eq!(y_cpu.len(), d_inner * n_tok);
        assert_eq!(s_cpu.len(), d_inner * d_state);

        // Print first few outputs
        println!("SSM Scan CPU output (first 8):");
        for i in 0..8.min(y_cpu.len()) {
            println!("  y[{}] = {}", i, y_cpu[i]);
        }
        println!("SSM Scan final state (first 8):");
        for i in 0..8.min(s_cpu.len()) {
            println!("  s[{}] = {}", i, s_cpu[i]);
        }

        // Verify no NaN
        for (i, &val) in y_cpu.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at y[{}]", i);
            assert!(!val.is_infinite(), "Inf at y[{}]", i);
        }
        for (i, &val) in s_cpu.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at s[{}]", i);
            assert!(!val.is_infinite(), "Inf at s[{}]", i);
        }
    }

    #[test]
    fn test_ssm_scan_with_initial_state() {
        // Test with non-zero initial state
        let d_inner = 4;
        let d_state = 16;
        let n_tok = 8;

        let x = vec![0.1f32; d_inner * n_tok];
        let dt = vec![0.001f32; d_inner * n_tok];
        let a = vec![-0.1f32; d_inner * d_state];
        let b = vec![0.05f32; d_state * n_tok];
        let c = vec![0.05f32; d_state * n_tok];
        let s0 = vec![1.0f32; d_inner * d_state]; // Non-zero initial state

        let (y_cpu, s_cpu) = ssm_scan_cpu(
            &x,
            &dt,
            &a,
            &b,
            &c,
            Some(&s0),
            d_inner,
            d_state,
            n_tok,
        );

        // Verify output is different from zero initial state
        let (y_zero, s_zero) =
            ssm_scan_cpu(&x, &dt, &a, &b, &c, None, d_inner, d_state, n_tok);

        // With positive initial state and negative A (decay), output should be
        // larger
        for i in 0..y_cpu.len() {
            assert!(
                y_cpu[i].abs() > y_zero[i].abs(),
                "Output with initial state should be larger at y[{}]: {} vs {}",
                i,
                y_cpu[i],
                y_zero[i]
            );
        }

        // Final state should also be larger
        for i in 0..s_cpu.len() {
            assert!(
                s_cpu[i].abs() > s_zero[i].abs(),
                "Final state with initial state should be larger at s[{}]",
                i
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_ssm_conv1d_gpu() {
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        // Test parameters
        let d_inner = 64;
        let d_conv = 4;
        let n_tok = 128;

        // Generate test data
        let mut x = Vec::with_capacity(d_inner * (n_tok + d_conv - 1));
        let mut w = Vec::with_capacity(d_inner * d_conv);
        let mut bias = Vec::with_capacity(d_inner);

        for i in 0..d_inner * (n_tok + d_conv - 1) {
            x.push((i as f32 * 0.01) - 1.0);
        }
        for i in 0..d_inner * d_conv {
            w.push((i as f32 * 0.05) - 0.1);
        }
        for i in 0..d_inner {
            bias.push(i as f32 * 0.01);
        }

        // CPU reference
        let y_cpu = ssm_conv1d_cpu(&x, &w, Some(&bias), d_inner, d_conv, n_tok);

        // GPU computation
        let ctx = MetalContext::new().expect("Failed to create Metal context");
        let y_gpu = ctx
            .ssm_conv1d_d4(&x, &w, Some(&bias), d_inner, n_tok)
            .expect("GPU SSM Conv1D failed");

        // Verify sizes match
        assert_eq!(y_cpu.len(), y_gpu.len(), "Output size mismatch");

        // Compare results
        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..y_cpu.len() {
            let diff = (y_cpu[i] - y_gpu[i]).abs();
            let rel = diff / y_cpu[i].abs().max(1e-10);
            max_diff = max_diff.max(diff);
            max_rel = max_rel.max(rel);
        }

        println!(
            "SSM Conv1D GPU vs CPU: max_diff={}, max_rel={}",
            max_diff, max_rel
        );

        // Allow small floating point differences
        // Note: max_rel can be large for values near zero
        // max_diff threshold of 1e-3 is reasonable for FP32
        assert!(
            max_diff < 1e-3,
            "GPU result differs from CPU: max_diff={}",
            max_diff
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_ssm_scan_gpu() {
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        // Test parameters
        let d_inner = 64;
        let d_state = 16;
        let n_tok = 128;

        // Generate test data
        let mut x = Vec::with_capacity(d_inner * n_tok);
        let mut dt = Vec::with_capacity(d_inner * n_tok);
        let mut a = Vec::with_capacity(d_inner * d_state);
        let mut b = Vec::with_capacity(d_state * n_tok);
        let mut c = Vec::with_capacity(d_state * n_tok);

        for i in 0..d_inner * n_tok {
            x.push((i as f32 * 0.01) - 0.5);
            dt.push((i as f32 * 0.001).abs() + 0.001); // Positive dt
        }
        for i in 0..d_inner * d_state {
            a.push(-((i % 10) as f32 + 1.0) * 0.1); // Negative A (decay)
        }
        for i in 0..d_state * n_tok {
            b.push((i as f32 * 0.01) - 0.3);
            c.push((i as f32 * 0.01) - 0.3);
        }

        // CPU reference
        let (y_cpu, s_cpu) =
            ssm_scan_cpu(&x, &dt, &a, &b, &c, None, d_inner, d_state, n_tok);

        // GPU computation
        let ctx = MetalContext::new().expect("Failed to create Metal context");
        let (y_gpu, s_gpu) = ctx
            .ssm_scan_d16(&x, &dt, &a, &b, &c, None, d_inner, n_tok)
            .expect("GPU SSM Scan failed");

        // Verify sizes match
        assert_eq!(y_cpu.len(), y_gpu.len(), "Y size mismatch");
        assert_eq!(s_cpu.len(), s_gpu.len(), "S size mismatch");

        // Compare results
        let mut max_y_diff = 0.0f32;
        let mut max_y_rel = 0.0f32;
        for i in 0..y_cpu.len() {
            let diff = (y_cpu[i] - y_gpu[i]).abs();
            let rel = diff / y_cpu[i].abs().max(1e-10);
            max_y_diff = max_y_diff.max(diff);
            max_y_rel = max_y_rel.max(rel);
        }

        let mut max_s_diff = 0.0f32;
        let mut max_s_rel = 0.0f32;
        for i in 0..s_cpu.len() {
            let diff = (s_cpu[i] - s_gpu[i]).abs();
            let rel = diff / s_cpu[i].abs().max(1e-10);
            max_s_diff = max_s_diff.max(diff);
            max_s_rel = max_s_rel.max(rel);
        }

        println!(
            "SSM Scan GPU vs CPU: y_max_diff={}, y_max_rel={}, s_max_diff={}, s_max_rel={}",
            max_y_diff, max_y_rel, max_s_diff, max_s_rel
        );

        // Allow small floating point differences
        assert!(
            max_y_rel < 1e-3,
            "GPU Y result differs from CPU: max_rel={}",
            max_y_rel
        );
        assert!(
            max_s_rel < 1e-3,
            "GPU S result differs from CPU: max_rel={}",
            max_s_rel
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn test_ssm_conv1d_cpu_stub() {
        // Stub test for non-aarch64 platforms
        let d_inner = 4;
        let d_conv = 4;
        let n_tok = 8;

        let x = vec![0.1f32; d_inner * (n_tok + d_conv - 1)];
        let w = vec![0.05f32; d_inner * d_conv];
        let bias = vec![0.01f32; d_inner];

        let y_cpu = ssm_conv1d_cpu(&x, &w, Some(&bias), d_inner, d_conv, n_tok);
        assert_eq!(y_cpu.len(), d_inner * n_tok);
    }

    #[test]
    fn test_ssm_scan_cpu_stub() {
        // Stub test for non-aarch64 platforms
        let d_inner = 4;
        let d_state = 16;
        let n_tok = 8;

        let x = vec![0.1f32; d_inner * n_tok];
        let dt = vec![0.001f32; d_inner * n_tok];
        let a = vec![-0.1f32; d_inner * d_state];
        let b = vec![0.05f32; d_state * n_tok];
        let c = vec![0.05f32; d_state * n_tok];

        let (y_cpu, s_cpu) =
            ssm_scan_cpu(&x, &dt, &a, &b, &c, None, d_inner, d_state, n_tok);
        assert_eq!(y_cpu.len(), d_inner * n_tok);
        assert_eq!(s_cpu.len(), d_inner * d_state);
    }
}
