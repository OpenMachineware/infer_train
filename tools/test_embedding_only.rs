// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test without any GPU operations - just embedding lookup

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::InferenceContext;
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        // Create one context and check embeddings multiple times
        let gpu_ctx = MetalContext::new().unwrap();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        for i in 1..=3 {
            println!("=== Test {}: Embedding Lookup ===", i);
            let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
            let nan_count = embedding.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "Embedding: {} NaN, first 5: {:?}",
                nan_count,
                &embedding[..5.min(embedding.len())]
            );
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
