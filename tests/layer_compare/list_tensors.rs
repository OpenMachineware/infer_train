// List all tensor names
use infer_train::model::GgufFile;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    println!("Total tensors: {}", gguf.tensors.len());

    // Find output-related tensors
    for tensor in &gguf.tensors {
        if tensor.name.starts_with("output") || tensor.name.starts_with("token_embd") || tensor.name.starts_with("lm_head") {
            println!("{}: {:?} dims={:?}", tensor.name, tensor.quant_type, tensor.dims);
        }
    }

    // Also print first 5 and last 5 tensors
    println!("\nFirst 5 tensors:");
    for tensor in &gguf.tensors[..5] {
        println!("  {}: {:?} dims={:?}", tensor.name, tensor.quant_type, tensor.dims);
    }
    println!("\nLast 5 tensors:");
    for tensor in &gguf.tensors[gguf.tensors.len()-5..] {
        println!("  {}: {:?} dims={:?}", tensor.name, tensor.quant_type, tensor.dims);
    }
}
