// Prefill performance test

use infer_train::inference::{ForwardExecutor, SimpleModel};
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    // Load model
    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");

    let mut executor = ForwardExecutor::new(model);

    let n_tokens = 512;
    let vocab_size = executor.model.config().vocab_size;
    let tokens: Vec<u32> =
        (0..n_tokens).map(|i| (i % (vocab_size - 1)) as u32 + 1).collect();

    // Warmup
    let _ = executor.forward_batch(&tokens);

    // Measure
    let n_runs = 5;
    let mut total_ns = 0u64;
    for _ in 0..n_runs {
        executor.reset();
        let start = Instant::now();
        let _ = executor.forward_batch(&tokens);
        total_ns += start.elapsed().as_nanos() as u64;
    }

    let avg_ns = total_ns / n_runs;
    let tps = 1e9 * n_tokens as f64 / avg_ns as f64;
    println!(
        "Prefill {} tokens: {:.2} t/s (avg {:.2} ms)",
        n_tokens,
        tps,
        avg_ns as f64 / 1e6
    );
}
