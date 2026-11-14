use infer_train::model::GgufFile;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    for layer in [0, 2, 27] {
        let tensor_name = format!("blk.{}.ffn_norm.weight", layer);
        if let Some(data) = gguf.get_tensor_data(&tensor_name) {
            let weight: Vec<f32> = data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            let min = weight.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = weight.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let abs_max = weight.iter().map(|x| x.abs()).fold(0.0, f32::max);

            // Count outliers
            let outliers: Vec<(usize, f32)> = weight.iter().enumerate()
                .filter(|(_, w)| w.abs() > 3.0)
                .map(|(i, w)| (i, *w))
                .collect();

            println!("{}: len={}, min={:.4}, max={:.4}, abs_max={:.4}",
                     tensor_name, weight.len(), min, max, abs_max);
            println!("  Outliers (|w| > 3.0): {} values", outliers.len());
            if !outliers.is_empty() {
                println!("  First 10: {:?}", &outliers[..outliers.len().min(10)]);
            }
        }
    }
}
