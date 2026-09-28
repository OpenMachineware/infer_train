// Minimal Flash Attention test with CPU reference
// Uses the simplest possible kernel to verify correctness

use infer_train::ops::gpu::metal::{FlashAttnMetalContext, KvFormat};
use std::time::Instant;

fn cpu_attention_reference(q: &[f32], k: &[f32], v: &[f32], n_heads: usize, head_dim: usize, seq_len: usize) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; n_heads * head_dim];
    
    for h in 0..n_heads {
        let q_offset = h * head_dim;
        
        // K and V are [n_heads, seq_len, head_dim]
        // K[h][s][d] = k[h * seq_len * head_dim + s * head_dim + d]
        
        // Compute attention scores and apply softmax
        let mut scores = vec![0.0f32; seq_len];
        let mut max_score = f32::NEG_INFINITY;
        
        for s in 0..seq_len {
            let k_offset = h * seq_len * head_dim + s * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_offset + d] * k[k_offset + d];
            }
            scores[s] = dot * scale;
            max_score = max_score.max(scores[s]);
        }
        
        // Softmax
        let mut sum_exp = 0.0f32;
        for s in 0..seq_len {
            scores[s] = (scores[s] - max_score).exp();
            sum_exp += scores[s];
        }
        
        // Weighted sum of V
        for d in 0..head_dim {
            let mut out = 0.0f32;
            for s in 0..seq_len {
                let v_offset = h * seq_len * head_dim + s * head_dim;
                out += scores[s] / sum_exp * v[v_offset + d];
            }
            output[h * head_dim + d] = out;
        }
    }
    
    output
}

fn main() {
    println!("=== Flash Attention Correctness Test ===");
    println!("Comparing GPU output with CPU reference\n");

    let ctx = match FlashAttnMetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    let n_heads = 32;
    let head_dim = 128;
    // Use seq_len that is a multiple of C=256 for the kernel
    // Otherwise, the kernel will access out-of-bounds memory
    let seq_len = 256;

    // Create simple test data with correct layout
    // Q layout: [n_heads, head_dim] - each head has one query
    let q: Vec<f32> = (0..n_heads * head_dim).map(|i| 0.5 + 0.01 * (i as f32 % 100.0)).collect();
    
    // K layout for the kernel: [n_heads, seq_len, head_dim]
    // The kernel accesses K[h, s, d] at offset h*nb12 + s*nb11 + d
    // where nb11 = head_dim * 2 (for F16) and nb12 = seq_len * nb11
    // So we need to organize K as [n_heads][seq_len][head_dim]
    let mut k: Vec<f32> = vec![0.0; n_heads * seq_len * head_dim];
    let mut v: Vec<f32> = vec![0.0; n_heads * seq_len * head_dim];
    for h in 0..n_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                let idx = h * seq_len * head_dim + s * head_dim + d;
                k[idx] = 0.3 + 0.01 * ((h * 1000 + s * 100 + d) as f32 % 100.0);
                v[idx] = 0.2 + 0.01 * ((h * 1000 + s * 100 + d) as f32 % 100.0);
            }
        }
    }

    // Convert to F16
    let k_f16: Vec<u16> = k.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    let v_f16: Vec<u16> = v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();

    // CPU reference
    println!("Computing CPU reference...");
    let cpu_output = cpu_attention_reference(&q, &k, &v, n_heads, head_dim, seq_len);
    println!("CPU output sample: {:?}", &cpu_output[0..4]);

    // GPU test
    println!("\nRunning GPU attention...");
    let k_bytes = unsafe { std::slice::from_raw_parts(k_f16.as_ptr() as *const u8, k_f16.len() * 2) };
    let v_bytes = unsafe { std::slice::from_raw_parts(v_f16.as_ptr() as *const u8, v_f16.len() * 2) };

    match ctx.flash_attn_decode(&q, k_bytes, v_bytes, None, n_heads as u32, n_heads as u32, head_dim as u32, seq_len as u32, KvFormat::F16) {
        Ok(gpu_output) => {
            println!("GPU output sample: {:?}", &gpu_output[0..4]);
            
            // Compare
            let mut max_diff = 0.0f32;
            let mut total_diff = 0.0f32;
            for i in 0..cpu_output.len() {
                let diff = (cpu_output[i] - gpu_output[i]).abs();
                max_diff = max_diff.max(diff);
                total_diff += diff;
            }
            let avg_diff = total_diff / cpu_output.len() as f32;
            
            println!("\nComparison:");
            println!("  Max difference: {:.6}", max_diff);
            println!("  Avg difference: {:.6}", avg_diff);
            
            if max_diff < 0.01 {
                println!("  ✓ PASS: Output matches CPU reference");
            } else {
                println!("  ✗ FAIL: Output differs from CPU reference");
            }
        }
        Err(e) => {
            eprintln!("GPU attention failed: {}", e);
        }
    }
}