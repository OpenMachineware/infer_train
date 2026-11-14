// Compare final logits between our implementation and reference
use infer_train::types::*;
use infer_train::model::GgufFile;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Get output norm and output projection weights
    let output_norm_name = "output_norm.weight";
    let output_norm_data = gguf.get_tensor_data(output_norm_name).unwrap();
    let output_norm_tensor = gguf.tensors.iter().find(|t| t.name == output_norm_name).unwrap();

    println!("=== Output Norm ===");
    println!("Type: {:?}", output_norm_tensor.quant_type);
    println!("Size: {}", output_norm_data.len());

    let output_weight_name = "output.weight";
    let output_weight_data = gguf.get_tensor_data(output_weight_name);
    let output_weight_tensor = gguf.tensors.iter().find(|t| t.name == output_weight_name).unwrap();

    println!("\n=== Output Projection ===");
    println!("Type: {:?}", output_weight_tensor.quant_type);
    println!("Dims: {:?}", output_weight_tensor.dims);
    if let Some(data) = output_weight_data {
        println!("Data size: {}", data.len());
    }

    // Check token embedding (should be tied with output projection)
    let token_embd_name = "token_embd.weight";
    let token_embd_data = gguf.get_tensor_data(token_embd_name).unwrap();
    let token_embd_tensor = gguf.tensors.iter().find(|t| t.name == token_embd_name).unwrap();

    println!("\n=== Token Embedding ===");
    println!("Type: {:?}", token_embd_tensor.quant_type);
    println!("Dims: {:?}", token_embd_tensor.dims);

    // Check if they share the same data
    if let Some(output_data) = output_weight_data {
        if token_embd_data.as_ptr() == output_data.as_ptr() {
            println!("TIED: token_embd and output.weight share the same data!");
        } else {
            println!("NOT TIED: Different data pointers");
        }
    }
}
