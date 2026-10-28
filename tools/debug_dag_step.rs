// Debug DAG executor step by step
// Compare each operation output between standard forward and DAG

use infer_train::dag::DagBuilder;
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

    // Standard forward
    let mut kv_std = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let logits_std = ctx.forward(&[test_token], &mut kv_std).unwrap();

    // Get embedding
    let hidden = ctx.model.embedding_lookup_impl(test_token).unwrap();
    println!("Embedding (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);

    // Get layer 0 components from standard forward
    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.config.hidden_size;

    // FFN norm
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();
    let normed = ctx.model.apply_rms_norm_impl(&hidden, ffn_norm, hidden_size).unwrap();
    println!("FFN norm output (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        normed[0], normed[1], normed[2], normed[3], normed[4]);

    // Gate projection (w1)
    let w1 = layer.w1.as_ref().unwrap();
    let intermediate_size = w1.shape[1];
    let gate = ctx.model.gemv_impl(&normed, w1, intermediate_size).unwrap();
    println!("Gate (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        gate[0], gate[1], gate[2], gate[3], gate[4]);

    // Up projection (w3)
    let w3 = layer.w3.as_ref().unwrap();
    let up = ctx.model.gemv_impl(&normed, w3, intermediate_size).unwrap();
    println!("Up (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        up[0], up[1], up[2], up[3], up[4]);

    // SiLU
    let silu: Vec<f32> = up.iter().map(|&u| u / (1.0 + (-u).exp())).collect();
    println!("SiLU (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        silu[0], silu[1], silu[2], silu[3], silu[4]);

    // Mul
    let hidden_ffn: Vec<f32> = gate.iter().zip(silu.iter()).map(|(g, s)| g * s).collect();
    println!("Hidden FFN (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        hidden_ffn[0], hidden_ffn[1], hidden_ffn[2], hidden_ffn[3], hidden_ffn[4]);

    // Down projection (w2) - this is transposed
    let w2 = layer.w2.as_ref().unwrap();
    println!("\nW2 shape: {:?}", w2.shape);
    println!("W2 dtype: {:?}", w2.dtype);

    // Using gemv_impl (wrong, treats it as normal)
    let ffn_out_wrong = ctx.model.gemv_impl(&hidden_ffn, w2, hidden_size).unwrap();
    println!("FFN out (wrong, using gemv_impl) (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        ffn_out_wrong[0], ffn_out_wrong[1], ffn_out_wrong[2], ffn_out_wrong[3], ffn_out_wrong[4]);

    // Using gemv_transposed (correct)
    let ffn_out_correct = ctx.model.gemv_transposed(&hidden_ffn, w2, hidden_size).unwrap();
    println!("FFN out (correct, using gemv_transposed) (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        ffn_out_correct[0], ffn_out_correct[1], ffn_out_correct[2], ffn_out_correct[3], ffn_out_correct[4]);

    // Difference
    let max_diff: f32 = ffn_out_wrong.iter().zip(ffn_out_correct.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("Max difference between wrong and correct: {:.4}", max_diff);

    println!("\nStandard forward logits (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        logits_std[0], logits_std[1], logits_std[2], logits_std[3], logits_std[4]);
}
