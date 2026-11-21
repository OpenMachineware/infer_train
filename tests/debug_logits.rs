//! Debug logits output
use infer_train::inference::forward::ForwardExecutor;
use infer_train::inference::simple_model::SimpleModel;
use infer_train::model::gguf::GGUFModel;
use infer_train::model::gguf::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let gguf = GgufFile::open(&args[1]).expect("Failed to load GGUF");
    let model = SimpleModel::load(gguf).expect("Failed to load model");
    let mut ctx = ForwardExecutor::new(model);

    // Tokenize "Hello"
    let tokens = ctx.model.tokenize("Hello").expect("Tokenization failed");
    println!("Tokens: {:?}", tokens);

    // Run inference for first token
    ctx.forward(tokens[0]).expect("Forward pass failed");

    // Run for remaining tokens
    for &token in &tokens[1..] {
        ctx.forward(token).expect("Forward pass failed");
    }

    // Run one more step to get logits
    let logits =
        ctx.forward(tokens[tokens.len() - 1]).expect("Final forward failed");

    // Find top 10 tokens
    let mut logit_pairs: Vec<(f32, usize)> =
        logits.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    logit_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("\nTop 10 tokens (our implementation):");
    for i in 0..10 {
        println!(
            "  {}: token={}, logit={:.6}",
            i, logit_pairs[i].1, logit_pairs[i].0
        );
    }

    // Specific tokens
    println!("\nSpecific tokens:");
    println!("  Token 0: logit={:.6}", logits[0]);
    println!("  Token 9707: logit={:.6}", logits[9707]);
    println!("  Token 151644: logit={:.6}", logits[151644]);
}
