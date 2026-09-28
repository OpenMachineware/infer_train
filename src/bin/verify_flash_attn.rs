use infer_train::ops::gpu::FlashAttnMetalContext;
use half::f16;

fn main() {
    println!("=== Flash Attention F16 Correctness Verification ===\n");

    let ctx = FlashAttnMetalContext::new().expect("Failed to create Metal context");

    // Test parameters - use small seq_len for debugging
    let head_dim = 128;
    let seq_len = 4;  // Small for debugging

    // Create simple test data
    let q: Vec<f16> = vec![f16::from_f32(1.0); head_dim];
    let k: Vec<f16> = vec![f16::from_f32(0.5); seq_len * head_dim];
    let v: Vec<f16> = (0..seq_len * head_dim)
        .map(|i| f16::from_f32((i / head_dim + 1) as f32))  // v[s] = (s+1)
        .collect();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Debug: compute scores manually
    println!("Debug: Q·K scores");
    let q_sum: f32 = q.iter().map(|x| x.to_f32()).sum();
    println!("  Q sum: {}", q_sum);
    let k0_sum: f32 = k[0..head_dim].iter().map(|x| x.to_f32()).sum();
    println!("  K[0] sum: {}", k0_sum);
    println!("  Expected score[0]: {} * {} * {} = {}", q_sum, k0_sum/head_dim as f32, scale, q_sum * (k0_sum/head_dim as f32) * scale);

    // Expected output
    println!("\nExpected softmax:");
    println!("  All scores equal = {}", 5.656854);
    println!("  max_val = 5.656854");
    println!("  weight for each = exp(0) = 1.0");
    println!("  sum_exp = 4.0");
    println!("  V values: [1.0, 2.0, 3.0, 4.0]");
    println!("  Expected output = (1*1 + 2*1 + 3*1 + 4*1) / 4 = 2.5");

    // GPU result
    let gpu_result = ctx.flash_attn_f16_decode(&q, &k, &v, None, head_dim, seq_len, scale)
        .expect("Flash attention failed");

    // CPU reference
    let cpu_result = cpu_flash_attn_f16(&q, &k, &v, head_dim, seq_len, scale);

    // Debug: show scores
    let mut cpu_scores: Vec<f32> = vec![0.0; seq_len];
    for s in 0..seq_len {
        let mut score = 0.0;
        for d in 0..head_dim {
            score += q[d].to_f32() * k[s * head_dim + d].to_f32();
        }
        cpu_scores[s] = score * scale;
    }
    let cpu_max = cpu_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  CPU max score: {}", cpu_max);
    println!("  CPU scores: {:?}", &cpu_scores);

    // Compare
    println!("\nComparing GPU vs CPU:");
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for i in 0..head_dim {
        let gpu_val = gpu_result[i].to_f32();
        let cpu_val = cpu_result[i];
        let diff = (gpu_val - cpu_val).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }

    println!("  Max diff: {}", max_diff);
    println!("  Avg diff: {}", sum_diff / head_dim as f32);
    println!("  First 5 values (GPU): {:?}", &gpu_result[..5.min(gpu_result.len())].iter().map(|x| x.to_f32()).collect::<Vec<_>>());
    println!("  First 5 values (CPU): {:?}", &cpu_result[..5.min(cpu_result.len())]);

    if max_diff < 0.001 {
        println!("\n✅ Results match within tolerance!");
    } else {
        println!("\n❌ Results differ significantly!");
    }

    // Test with causal mask
    println!("\n=== Testing with causal mask ===");

    let mask: Vec<f16> = (0..seq_len)
        .map(|i| if i < seq_len / 2 { f16::from_f32(-f32::INFINITY) } else { f16::ZERO })
        .collect();

    let gpu_masked = ctx.flash_attn_f16_decode(&q, &k, &v, Some(&mask), head_dim, seq_len, scale)
        .expect("Flash attention failed");

    let cpu_masked = cpu_flash_attn_f16_masked(&q, &k, &v, &mask, head_dim, seq_len, scale);

    let mut max_diff_masked = 0.0f32;
    for i in 0..head_dim {
        let diff = (gpu_masked[i].to_f32() - cpu_masked[i]).abs();
        max_diff_masked = max_diff_masked.max(diff);
    }

    println!("  Max diff (masked): {}", max_diff_masked);

    if max_diff_masked < 0.001 {
        println!("  ✅ Masked results match!");
    } else {
        println!("  ❌ Masked results differ!");
    }
}

fn cpu_flash_attn_f16(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    let mut sum_exp = 0.0;
    let mut output = vec![0.0; head_dim];

    // Compute scores and find max
    let mut scores = vec![0.0; seq_len];
    for s in 0..seq_len {
        let mut score = 0.0;
        for d in 0..head_dim {
            score += q[d].to_f32() * k[s * head_dim + d].to_f32();
        }
        scores[s] = score * scale;
        if scores[s] > max_val {
            max_val = scores[s];
        }
    }

    // Softmax and accumulate
    for s in 0..seq_len {
        scores[s] = (scores[s] - max_val).exp();
        sum_exp += scores[s];
    }

    for s in 0..seq_len {
        let weight = scores[s] / sum_exp;
        for d in 0..head_dim {
            output[d] += weight * v[s * head_dim + d].to_f32();
        }
    }

    output
}

fn cpu_flash_attn_f16_masked(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    mask: &[f16],
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Vec<f32> {
    let mut max_val = f32::NEG_INFINITY;
    let mut sum_exp = 0.0;
    let mut output = vec![0.0; head_dim];

    let mut scores = vec![0.0; seq_len];
    for s in 0..seq_len {
        if mask[s].to_f32() == f32::NEG_INFINITY {
            scores[s] = f32::NEG_INFINITY;
        } else {
            let mut score = 0.0;
            for d in 0..head_dim {
                score += q[d].to_f32() * k[s * head_dim + d].to_f32();
            }
            scores[s] = score * scale;
        }
        if scores[s] > max_val {
            max_val = scores[s];
        }
    }

    for s in 0..seq_len {
        if scores[s] != f32::NEG_INFINITY {
            scores[s] = (scores[s] - max_val).exp();
            sum_exp += scores[s];
        }
    }

    for s in 0..seq_len {
        if scores[s] != f32::NEG_INFINITY {
            let weight = scores[s] / sum_exp;
            for d in 0..head_dim {
                output[d] += weight * v[s * head_dim + d].to_f32();
            }
        }
    }

    output
}
