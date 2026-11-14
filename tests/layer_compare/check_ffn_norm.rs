// Check FFN norm weights across all layers
use infer_train::model::GgufFile;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    for layer_idx in 0..28 {
        let tensor_name = format!("blk.{}.ffn_norm.weight", layer_idx);
        if let Some(data) = gguf.get_tensor_data(&tensor_name) {
            let tensor = gguf.tensors.iter().find(|t| t.name == tensor_name).unwrap();

            // Parse as F32
            let weights: Vec<f32> = data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;

            // Check for outliers
            let outliers: Vec<f32> = weights.iter()
                .filter(|&&w| w.abs() > 2.0)
                .cloned()
                .collect();

            println!("L{}: min={:.4} max={:.4} mean={:.4} outliers={} quant={:?}",
                     layer_idx, min, max, mean, outliers.len(), tensor.quant_type);
            if !outliers.is_empty() {
                println!("  outlier values: {:?}", &outliers[..outliers.len().min(10)]);
            }
        }
    }
}
