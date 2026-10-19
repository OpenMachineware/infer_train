// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check buffer allocations per forward pass

fn main() {
    println!("=== Buffer Allocations Per Forward Pass ===\n");

    // From forward_layer_gpu_encode
    println!("Per layer:");
    println!("  normed_buffer: hidden_size × 4 bytes");
    println!("  q_buffer: q_out_size × 4 bytes");
    println!("  k_buffer: k_out_size × 4 bytes");
    println!("  v_buffer: v_out_size × 4 bytes");
    println!("  q_rope_buffer: q_out_size × 4 bytes");
    println!("  k_rope_buffer: k_out_size × 4 bytes");
    println!("  attn_out_buffer: q_out_size × 4 bytes");
    println!("  attn_proj_buffer: hidden_size × 4 bytes");
    println!("  residual_buffer: hidden_size × 4 bytes");
    println!("  ffn_normed_buffer: hidden_size × 4 bytes");
    println!("  gate_buffer: intermediate_size × 4 bytes");
    println!("  up_buffer: intermediate_size × 4 bytes");
    println!("  silu_buffer: intermediate_size × 4 bytes");
    println!("  hidden_ffn_buffer: intermediate_size × 4 bytes");
    println!("  ffn_out_buffer: hidden_size × 4 bytes");
    println!("  final_buffer: hidden_size × 4 bytes");

    println!("\nTotal: 16 buffers per layer");

    // Calculate total memory
    let hidden_size = 1024;
    let intermediate_size = 3072;
    let q_out_size = 2048;
    let k_out_size = 1024;

    let per_layer_bytes =
        hidden_size * 4 +      // normed
        q_out_size * 4 +       // q
        k_out_size * 4 +       // k
        k_out_size * 4 +       // v
        q_out_size * 4 +       // q_rope
        k_out_size * 4 +       // k_rope
        q_out_size * 4 +       // attn_out
        hidden_size * 4 +      // attn_proj
        hidden_size * 4 +      // residual
        hidden_size * 4 +      // ffn_normed
        intermediate_size * 4 + // gate
        intermediate_size * 4 + // up
        intermediate_size * 4 + // silu
        intermediate_size * 4 + // hidden_ffn
        hidden_size * 4 +      // ffn_out
        hidden_size * 4;       // final

    println!("\nFor Qwen3-0.6B (hidden=1024, intermediate=3072):");
    println!("  Per layer: {} KB", per_layer_bytes / 1024);
    println!("  28 layers: {} KB", per_layer_bytes * 28 / 1024);

    println!("\nPlus final norm + output:");
    println!("  final_hidden_buffer: {} KB", hidden_size * 4 / 1024);
    println!("  logits_buffer: {} KB", 151936 * 4 / 1024);

    println!("\nTotal GPU memory per forward pass: {} KB",
        (per_layer_bytes * 28 + hidden_size * 4 + 151936 * 4) / 1024);

    println!("\n=== Optimization Opportunities ===");
    println!("1. Reuse buffers across layers (but sequential dependency)");
    println!("2. Use smaller dtypes (F16 instead of F32) for intermediates");
    println!("3. In-place operations where possible");
}
