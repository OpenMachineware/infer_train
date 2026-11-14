// Debug logits
use infer_train::inference::ForwardExecutor;
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");
    
    // Create tokenizer
    let meta = gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    let tokenizer = create_tokenizer(meta).expect("Failed to create tokenizer");
    
    // Create model
    let model = infer_train::inference::SimpleModel::load(model_path, tokenizer).expect("Failed to create model");
    
    // Create executor
    let mut executor = ForwardExecutor::new(model);
    
    // Process the first token only
    let token_id = 27u32;  // <|im_start|>
    
    println!("\n=== Processing token {} ===", token_id);
    let logits = executor.forward(token_id).expect("Forward failed");
    
    println!("Logits len: {}", logits.len());
    
    // Find top tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 20 tokens:");
    for (i, (idx, logit)) in indexed[..20].iter().enumerate() {
        println!("  {}: token {} logit {:.4}", i, idx, logit);
    }
    
    println!("\nBottom 20 tokens:");
    for (i, (idx, logit)) in indexed[indexed.len()-20..].iter().enumerate() {
        println!("  {}: token {} logit {:.4}", i, idx, logit);
    }
    
    // Compute softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    
    println!("\nSoftmax stats:");
    println!("  max_logit: {:.4}", max_logit);
    println!("  exp_sum: {:.4}", exp_sum);
    
    // Top softmax probabilities
    let mut probs: Vec<(usize, f32)> = logits.iter()
        .map(|&l| ((l - max_logit).exp() / exp_sum))
        .enumerate()
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 20 probabilities:");
    for (i, (idx, prob)) in probs[..20].iter().enumerate() {
        println!("  {}: token {} prob {:.6}", i, idx, prob);
    }
}