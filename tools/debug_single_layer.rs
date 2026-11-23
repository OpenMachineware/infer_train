// Debug Q/K norm weights and their effect
use infer_train::inference::{ForwardExecutor, SimpleModel};
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or(
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf".to_string(),
    );
    let gguf = GgufFile::load(&model_path).expect("Failed to load GGUF");
    let tokenizer_meta =
        gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    let tokenizer =
        create_tokenizer(tokenizer_meta).expect("Failed to create tokenizer");

    // Get Q and K norm weights
    let q_norm_weights = gguf
        .get_tensor_data("blk.0.attn_q_norm.weight")
        .expect("Q norm weight");
    let k_norm_weights = gguf
        .get_tensor_data("blk.0.attn_k_norm.weight")
        .expect("K norm weight");

    let q_norm_w: Vec<f32> = q_norm_weights
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let k_norm_w: Vec<f32> = k_norm_weights
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Print weight stats
    println!("=== Q Norm Weights ===");
    let q_mean: f32 = q_norm_w.iter().sum::<f32>() / q_norm_w.len() as f32;
    let q_max = q_norm_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("len={}, mean={:.4}, max={:.4}", q_norm_w.len(), q_mean, q_max);
    println!("First 10: {:?}", &q_norm_w[..10]);

    println!("\n=== K Norm Weights ===");
    let k_mean: f32 = k_norm_w.iter().sum::<f32>() / k_norm_w.len() as f32;
    let k_max = k_norm_w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let k_min = k_norm_w.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "len={}, mean={:.4}, min={:.4}, max={:.4}",
        k_norm_w.len(),
        k_mean,
        k_min,
        k_max
    );

    // Find the index of the max value
    let max_idx = k_norm_w
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i);
    println!("Max value index: {:?}", max_idx);
    println!("Max value: {:?}", max_idx.map(|i| k_norm_w[i]));

    // Print all large weights (> 10)
    println!("\nLarge K norm weights (> 10):");
    for (i, &w) in k_norm_w.iter().enumerate() {
        if w > 10.0 {
            println!("  index {}: {:.4}", i, w);
        }
    }

    println!("\nFirst 10: {:?}", &k_norm_w[..10]);
    println!("Last 10: {:?}", &k_norm_w[k_norm_w.len() - 10..]);
}
