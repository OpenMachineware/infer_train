use infer_train::inference::ForwardExecutor;
use infer_train::inference::SimpleModel;
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    let meta =
        gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    let tokenizer = create_tokenizer(meta).expect("Failed to create tokenizer");

    // Test token 45451 (which was top token in debug output)
    let token_id = 45451;
    let token_str = tokenizer.decode(&[token_id]);
    println!("Token {}: {:?}", token_id, token_str);

    // Test decode of some tokens
    for id in [0, 1, 100, 1000, 45451, 102224] {
        let result = tokenizer.decode(&[id]);
        println!("Token {}: {:?}", id, result);
    }

    // Check if tokenizer has raw token strings
    println!("\nChecking raw token data...");
}
