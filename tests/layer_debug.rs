// Debug layer by layer to find the garbled output source
use infer_train::inference::{ForwardExecutor, SimpleModel};
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

#[test]
fn layer_debug_test() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");
    let tokenizer_meta =
        gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    let tokenizer =
        create_tokenizer(tokenizer_meta).expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");
    let mut executor = ForwardExecutor::new(model);
    executor.debug_mode = true;

    // Simple input: "Hello" -> token 15286
    let tokens = vec![15286u32];

    println!("=== Layer-by-Layer Debug ===");
    println!("Input tokens: {:?}", tokens);

    // Run forward and collect debug states
    match executor.forward_batch(&tokens) {
        Ok(logits) => {
            println!("\n=== Final Logits ===");
            println!("Logits shape: {}", logits.len());

            // Get top 5 tokens
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("Top 5 tokens:");
            for (idx, val) in indexed.iter().take(5) {
                println!("  token {} : {:.4}", idx, val);
            }

            // Print debug states for each layer
            println!("\n=== Debug States ===");
            for state in &executor.debug_states {
                let hidden_mean = state.hidden_after.iter().sum::<f32>()
                    / state.hidden_after.len() as f32;
                let hidden_std = {
                    let variance: f32 = state
                        .hidden_after
                        .iter()
                        .map(|x| (x - hidden_mean).powi(2))
                        .sum::<f32>()
                        / state.hidden_after.len() as f32;
                    variance.sqrt()
                };

                let q_mean = state.q.iter().sum::<f32>() / state.q.len() as f32;
                let k_mean = state.k.iter().sum::<f32>() / state.k.len() as f32;
                let v_mean = state.v.iter().sum::<f32>() / state.v.len() as f32;
                let attn_out_mean = state.attn_out.iter().sum::<f32>()
                    / state.attn_out.len() as f32;
                let ffn_out_mean = state.ffn_out.iter().sum::<f32>()
                    / state.ffn_out.len() as f32;

                println!("Layer {}:", state.layer_idx);
                println!(
                    "  hidden_after: mean={:.6}, std={:.6}",
                    hidden_mean, hidden_std
                );
                println!(
                    "  q: mean={:.6}, k: mean={:.6}, v: mean={:.6}",
                    q_mean, k_mean, v_mean
                );
                println!(
                    "  attn_out: mean={:.6}, ffn_out: mean={:.6}",
                    attn_out_mean, ffn_out_mean
                );
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}
