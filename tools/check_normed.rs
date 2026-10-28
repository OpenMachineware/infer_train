// Check if normed has non-zero values
use infer_train::inference::Device;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];
    let ctx = infer_train::inference::InferenceContext::from_gguf(
        model_path,
        Device::Cpu,
    )
    .unwrap();

    let layer = &ctx.model.layers[0];
    let hidden = ctx.model.embedding_lookup_impl(1u32).unwrap();
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();
    let normed = ctx
        .model
        .apply_rms_norm_impl(&hidden, ffn_norm, ctx.config.hidden_size)
        .unwrap();

    // Check non-zero count
    let non_zero = normed.iter().filter(|&&x| x != 0.0).count();
    println!("normed non-zero: {}/{}", non_zero, normed.len());

    // Check first and last values
    println!(
        "First 10: {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
        normed[0],
        normed[1],
        normed[2],
        normed[3],
        normed[4],
        normed[5],
        normed[6],
        normed[7],
        normed[8],
        normed[9]
    );
    println!(
        "Last 10: {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
        normed[normed.len() - 10],
        normed[normed.len() - 9],
        normed[normed.len() - 8],
        normed[normed.len() - 7],
        normed[normed.len() - 6],
        normed[normed.len() - 5],
        normed[normed.len() - 4],
        normed[normed.len() - 3],
        normed[normed.len() - 2],
        normed[normed.len() - 1]
    );
}
