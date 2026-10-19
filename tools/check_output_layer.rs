// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check output layer properties

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::InferenceContext;
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let gpu_ctx = MetalContext::new().unwrap();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        if let Some(output) = &ctx.model().output {
            println!("Output layer:");
            println!("  shape: {:?}", output.shape);
            println!("  dtype: {:?}", output.dtype);
            println!("  elem_count: {}", output.elem_count());
            println!("  bytes_len: {}", output.bytes().len());
            println!(
                "  is_transposed: {}",
                output.shape.len() == 2
                    && output.shape[1] == ctx.model().config.hidden_size
            );
        }
    }
}
