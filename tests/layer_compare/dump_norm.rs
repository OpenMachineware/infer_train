// Dump FFN norm weights
use infer_train::model::GgufFile;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    for layer_idx in [0, 1, 2, 7] {
        let tensor_name = format!("blk.{}.ffn_norm.weight", layer_idx);
        println!("\n=== {} ===", tensor_name);

        if let Some(data) = gguf.get_tensor_data(&tensor_name) {
            let tensor = gguf.tensors.iter().find(|t| t.name == tensor_name).unwrap();
            println!("Quant type: {:?}", tensor.quant_type);
            println!("Dims: {:?}", tensor.dims);
            println!("Data len: {} bytes", data.len());

            match tensor.quant_type {
                infer_train::types::QuantType::F32 => {
                    let weights: &[f32] = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                    };
                    let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = weights.iter().sum();
                    println!("Weights (F32): min={:.4} max={:.4} sum={:.4} len={}", min, max, sum, weights.len());
                    println!("First 10: {:?}", &weights[..10.min(weights.len())]);
                    println!("Last 10: {:?}", &weights[weights.len().saturating_sub(10)..]);
                }
                _ => {
                    println!("Unsupported quant type");
                }
            }
        }
    }
}
