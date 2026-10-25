// Check vocab_size vs output shape
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = &args[1];

    let ctx = infer_train::inference::InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    ).unwrap();

    println!("Config vocab_size: {}", ctx.config.vocab_size);
    println!("Config hidden_size: {}", ctx.config.hidden_size);

    if let Some(output) = &ctx.model.output {
        println!("\nOutput tensor:");
        println!("  dtype: {:?}", output.dtype);
        println!("  shape: {:?}", output.shape);
        println!("  Expected shape: [{}, {}]", ctx.config.vocab_size, ctx.config.hidden_size);
    }
}
