// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Analyze GEMV operation sizes in the model

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::InferenceContext;
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let gpu_ctx = MetalContext::new().unwrap();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        println!("=== Model GEMV Operation Sizes ===\n");

        // Analyze each layer
        for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
            println!("Layer {}:", layer_idx);

            // QKV projections
            if let Some(wq) = layer.wq.as_ref() {
                let m = wq.shape[0];
                let k = wq.shape[1];
                let ops = m as f64 * k as f64 * 2.0;
                println!(
                    "  wq: {} x {} ({:.2}M ops) {:?}",
                    m,
                    k,
                    ops / 1e6,
                    wq.dtype
                );
            }
            if let Some(wk) = layer.wk.as_ref() {
                let m = wk.shape[0];
                let k = wk.shape[1];
                let ops = m as f64 * k as f64 * 2.0;
                println!("  wk: {} x {} ({:.2}M ops)", m, k, ops / 1e6);
            }
            if let Some(wv) = layer.wv.as_ref() {
                let m = wv.shape[0];
                let k = wv.shape[1];
                let ops = m as f64 * k as f64 * 2.0;
                println!("  wv: {} x {} ({:.2}M ops)", m, k, ops / 1e6);
            }
            if let Some(wo) = layer.wo.as_ref() {
                let m = wo.shape[0];
                let k = wo.shape[1];
                let ops = m as f64 * k as f64 * 2.0;
                println!("  wo: {} x {} ({:.2}M ops)", m, k, ops / 1e6);
            }

            // FFN projections
            if let Some(w1) = layer.w1.as_ref() {
                let m = w1.shape[0];
                let k = w1.shape[1];
                let ops = m as f64 * k as f64 * 2.0;
                println!("  w1 (gate): {} x {} ({:.2}M ops)", m, k, ops / 1e6);
            }
            if let Some(w3) = layer.w3.as_ref() {
                let m = w3.shape[0];
                let k = w3.shape[1];
                let ops = m as f64 * k as f64 * 2.0;
                println!("  w3 (up): {} x {} ({:.2}M ops)", m, k, ops / 1e6);
            }
            if let Some(w2) = layer.w2.as_ref() {
                let m = w2.shape[0];
                let k = w2.shape[1];
                let ops = m as f64 * k as f64 * 2.0;
                println!("  w2 (down): {} x {} ({:.2}M ops)", m, k, ops / 1e6);
            }

            println!();
        }

        // Output projection
        println!("Output layer:");
        if let Some(output) = ctx.model.output.as_ref() {
            let m = output.shape[0];
            let k = output.shape[1];
            let ops = m as f64 * k as f64 * 2.0;
            println!("  output: {} x {} ({:.2}M ops)", m, k, ops / 1e6);
        }
    }
}
