// Trace FFN operations step by step
// Compare standard forward vs DAG executor

use infer_train::inference::{Device, InferenceContext, KVCache, KVCacheFormat};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let test_token = 1u32;

    // Get embedding
    let hidden = ctx.model.embedding_lookup_impl(test_token).unwrap();
    println!("=== Step-by-step FFN trace ===\n");
    println!("Embedding (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);

    // Get layer 0
    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.config.hidden_size;

    // Attention norm
    let attn_norm = layer.attention_norm.as_ref().unwrap();
    let attn_normed = ctx.model.apply_rms_norm_impl(&hidden, attn_norm, hidden_size).unwrap();
    println!("After attention norm (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        attn_normed[0], attn_normed[1], attn_normed[2], attn_normed[3], attn_normed[4]);

    // Q/K/V projections
    let wq = layer.wq.as_ref().unwrap();
    let q_size = wq.shape[1];
    let q = ctx.model.gemv_impl(&attn_normed, wq, q_size).unwrap();
    println!("Q (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        q[0], q[1], q[2], q[3], q[4]);

    // Skip to FFN - use residual from embedding
    let mut residual = hidden.clone();

    // FFN norm
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();
    let ffn_normed = ctx.model.apply_rms_norm_impl(&residual, ffn_norm, hidden_size).unwrap();
    println!("\nFFN normed (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        ffn_normed[0], ffn_normed[1], ffn_normed[2], ffn_normed[3], ffn_normed[4]);

    // Gate projection (w1)
    let w1 = layer.w1.as_ref().unwrap();
    let intermediate_size = w1.shape[1];
    let gate = ctx.model.gemv_impl(&ffn_normed, w1, intermediate_size).unwrap();
    println!("Gate/w1 (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        gate[0], gate[1], gate[2], gate[3], gate[4]);

    // Up projection (w3)
    let w3 = layer.w3.as_ref().unwrap();
    let up = ctx.model.gemv_impl(&ffn_normed, w3, intermediate_size).unwrap();
    println!("Up/w3 (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        up[0], up[1], up[2], up[3], up[4]);

    // SiLU
    let silu: Vec<f32> = up.iter().map(|&u| u / (1.0 + (-u).exp())).collect();
    println!("SiLU (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        silu[0], silu[1], silu[2], silu[3], silu[4]);

    // Mul
    let hidden_ffn: Vec<f32> = gate.iter().zip(silu.iter()).map(|(g, s)| g * s).collect();
    println!("Gate * SiLU (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        hidden_ffn[0], hidden_ffn[1], hidden_ffn[2], hidden_ffn[3], hidden_ffn[4]);

    // Down projection (w2) - test both methods
    let w2 = layer.w2.as_ref().unwrap();
    println!("\nW2 info:");
    println!("  Shape: {:?}", w2.shape);
    println!("  Dtype: {:?}", w2.dtype);

    // Using gemv_impl (wrong)
    let ffn_wrong = ctx.model.gemv_impl(&hidden_ffn, w2, hidden_size).unwrap();
    println!("\nFFN out (gemv_impl - WRONG) (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        ffn_wrong[0], ffn_wrong[1], ffn_wrong[2], ffn_wrong[3], ffn_wrong[4]);

    // Using gemv_transposed (correct)
    let ffn_correct = ctx.model.gemv_transposed(&hidden_ffn, w2, hidden_size).unwrap();
    println!("FFN out (gemv_transposed - CORRECT) (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        ffn_correct[0], ffn_correct[1], ffn_correct[2], ffn_correct[3], ffn_correct[4]);

    // Compare difference
    let diff: f32 = ffn_wrong.iter().zip(ffn_correct.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("Sum of absolute differences: {:.4}", diff);

    // Now run full forward to see final output
    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits = ctx.forward(&[test_token], &mut kv).unwrap();
    println!("\nFull forward logits (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        logits[0], logits[1], logits[2], logits[3], logits[4]);
}
