// SPDX-License-Identifier: AGPL-3.0-or-later
use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let ctx = InferenceContext::from_gguf(&args[1], Device::Cpu)
        .expect("Failed to load model");

    // Check output tensor
    let output = ctx.model.output.as_ref().expect("No output tensor");
    println!("Output tensor: dtype={:?}, shape={:?}", output.dtype, output.shape);

    // Try a simple GEMV with random input
    let hidden_size = ctx.config.hidden_size;
    let vocab_size = ctx.config.vocab_size;

    let input: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.001).collect();

    println!("Testing GEMV with input len={}", input.len());
    let result = ctx.model.gemv(&input, output, vocab_size);

    match result {
        Some(logits) => {
            let sum: f32 = logits.iter().sum();
            let non_zero = logits.iter().filter(|&&x| x != 0.0).count();
            println!("GEMV result: len={}, sum={:.4}, non_zero={}, first 5: {:?}",
                logits.len(), sum, non_zero, &logits[..5.min(logits.len())]);
        }
        None => {
            println!("GEMV returned None!");
        }
    }
}
