// Check what tokens correspond to specific IDs
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

fn main() {
    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");
    let tokenizer_meta = gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    let tokenizer = create_tokenizer(tokenizer_meta).expect("Failed to create tokenizer");

    // Check specific token IDs
    let token_ids: Vec<u32> = vec![963, 63367, 103001, 64559, 16095, 110, 42379, 28319, 129816, 101766];

    println!("=== Token ID to String Mapping ===");
    for id in token_ids {
        match tokenizer.decode(&[id]) {
            Ok(s) => println!("Token {}: {:?}", id, s),
            Err(e) => println!("Token {}: Error decoding: {}", id, e),
        }
    }

    // Also check what "Hello" should generate
    println!("\n=== Expected response for 'Hello' ===");
    println!("With llama.cpp, the expected response is:");
    println!("Okay, the user said \"Hello\". I need to respond appropriately. Since I'm a");
    println!("\nFirst few tokens should be something like: ' Okay' or ',' etc.");

    // Check if token 963 is a common starting token
    println!("\nToken 963 decoding:");
    match tokenizer.decode(&[963]) {
        Ok(s) => println!("  {:?}", s),
        Err(e) => println!("  Error: {}", e),
    }
}
