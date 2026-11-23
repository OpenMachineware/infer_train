// Debug first token output
use infer_train::inference::{ForwardExecutor, SimpleModel};
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

#[test]
fn debug_first_token_test() {
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

    // Just one token: "Hello" -> token 15286
    let tokens = vec![15286u32];

    println!("=== Debug First Token ===");
    println!("Input token: {}", tokens[0]);

    match executor.forward_batch(&tokens) {
        Ok(logits) => {
            println!("Logits shape: {}", logits.len());

            // Get top 10 tokens
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("\nTop 10 tokens:");
            for (idx, val) in indexed.iter().take(10) {
                println!("  token {} : {:.4}", idx, val);
            }

            // Expected: token 30 ("[") for Qwen3 thinking mode
            println!("\nExpected token 30 ('[') logits: {:.4}", logits[30]);
            println!("Actual top token: {} ({})", indexed[0].0, indexed[0].1);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}
