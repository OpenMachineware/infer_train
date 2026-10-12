// Test GGUF parsing
// Run: cargo run --release --bin test_gguf -- models/qwen-7b-q4_0.gguf

use infer_train::model::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-file>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    println!("Loading GGUF file: {}", path);

    match GgufFile::open(path) {
        Ok(gguf) => {
            println!("✓ GGUF version: {}", gguf.version);
            println!("✓ Tensor count: {}", gguf.tensor_count);
            println!("✓ Metadata count: {}", gguf.metadata_kv_count);
            println!();

            // Print some metadata
            println!("=== Key Metadata ===");
            for (key, value) in &gguf.metadata {
                if key.starts_with("general.") || key.starts_with("model.") {
                    println!("  {}: {:?}", key, value);
                }
            }
            println!();

            // Print first few tensors
            println!("=== Tensors (first 10) ===");
            for tensor in gguf.tensors.iter().take(10) {
                println!(
                    "  {} {:?} {:?}",
                    tensor.name, tensor.dims, tensor.dtype
                );
            }

            // Verify tensor data is accessible
            if let Some(first_tensor) = gguf.tensors.first() {
                if let Some(data) = gguf.get_tensor_data(&first_tensor.name) {
                    println!(
                        "\n✓ First tensor data accessible: {} bytes",
                        data.len()
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            std::process::exit(1);
        }
    }
}
