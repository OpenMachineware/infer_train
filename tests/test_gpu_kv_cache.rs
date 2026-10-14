// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GPU KV cache and Flash Attention integration

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod tests {
    use infer_train::inference::KVCacheGPU;
    use infer_train::model::config::ModelConfig;
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use metal::MTLResourceOptions;
    use std::collections::HashMap;

    fn create_test_config() -> ModelConfig {
        ModelConfig {
            architecture: "llama".to_string(),
            hidden_size: 4096,
            intermediate_size: 11008,
            n_layers: 2,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            bos_token_id: 1,
            eos_token_id: 2,
            rms_norm_eps: 1e-5,
            rope_dim: 128,
            rope_freq_base: 10000.0,
            max_position_embeddings: 2048,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_gpu_kv_cache_creation() {
        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let config = create_test_config();
        let max_seq = 256;

        let kv_cache = KVCacheGPU::new(ctx.device(), &config, max_seq);

        assert_eq!(kv_cache.n_kv_heads(), config.n_kv_heads);
        assert_eq!(kv_cache.head_dim(), config.head_dim);
        assert_eq!(kv_cache.max_seq(), max_seq);
        assert_eq!(kv_cache.seq_len(), 0);
    }

    #[test]
    fn test_gpu_kv_cache_set_get() {
        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let config = create_test_config();
        let max_seq = 256;

        let mut kv_cache = KVCacheGPU::new(ctx.device(), &config, max_seq);

        // Create test K/V values
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;
        let k: Vec<f32> =
            (0..n_kv_heads * head_dim).map(|i| i as f32 * 0.01).collect();
        let v: Vec<f32> =
            (0..n_kv_heads * head_dim).map(|i| i as f32 * 0.02).collect();

        // Set K/V at position 0
        kv_cache.set_k(0, &k);
        kv_cache.set_v(0, &v);

        // Get back and verify
        let k_back = kv_cache.get_k(0, 0);
        let v_back = kv_cache.get_v(0, 0);

        // Allow small precision loss due to F16 conversion
        for i in 0..k.len() {
            let diff = (k[i] - k_back[i]).abs();
            assert!(
                diff < 0.01,
                "K mismatch at {}: expected {}, got {}, diff {}",
                i,
                k[i],
                k_back[i],
                diff
            );
        }
        for i in 0..v.len() {
            let diff = (v[i] - v_back[i]).abs();
            assert!(
                diff < 0.01,
                "V mismatch at {}: expected {}, got {}, diff {}",
                i,
                v[i],
                v_back[i],
                diff
            );
        }
    }

    #[test]
    fn test_gpu_kv_cache_advance() {
        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let config = create_test_config();
        let max_seq = 256;

        let mut kv_cache = KVCacheGPU::new(ctx.device(), &config, max_seq);

        assert_eq!(kv_cache.seq_len(), 0);

        kv_cache.advance();
        assert_eq!(kv_cache.seq_len(), 1);

        kv_cache.advance();
        assert_eq!(kv_cache.seq_len(), 2);
    }

    #[test]
    fn test_gpu_kv_cache_clear() {
        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let config = create_test_config();
        let max_seq = 256;

        let mut kv_cache = KVCacheGPU::new(ctx.device(), &config, max_seq);

        // Set some values
        let k = vec![1.0f32; config.n_kv_heads * config.head_dim];
        kv_cache.set_k(0, &k);
        kv_cache.advance();

        // Clear
        kv_cache.clear();

        assert_eq!(kv_cache.seq_len(), 0);

        // Verify buffers are zeroed
        let k_back = kv_cache.get_k(0, 0);
        for v in k_back {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_flash_attention_with_gpu_kv_cache() {
        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let config = create_test_config();
        let max_seq = 256;

        let mut kv_cache = KVCacheGPU::new(ctx.device(), &config, max_seq);

        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;

        // Create test Q
        let q: Vec<f32> =
            (0..n_heads * head_dim).map(|i| i as f32 * 0.001).collect();

        // Create test K/V and store in cache
        let k: Vec<f32> =
            (0..n_kv_heads * head_dim).map(|i| i as f32 * 0.001).collect();
        let v: Vec<f32> =
            (0..n_kv_heads * head_dim).map(|i| i as f32 * 0.002).collect();

        kv_cache.set_k(0, &k);
        kv_cache.set_v(0, &v);
        kv_cache.advance();

        // Run Flash Attention
        let k_buffer = kv_cache.k_buffer(0);
        let v_buffer = kv_cache.v_buffer(0);

        let q_buffer = std::rc::Rc::new(ctx.device().new_buffer_with_data(
            q.as_ptr() as *const std::ffi::c_void,
            (q.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let result = ctx.flash_attn_f16_gpu(
            1, // n_seqs
            n_heads, n_kv_heads, head_dim, 1, // seq_len
            &q_buffer, k_buffer, v_buffer,
        );

        assert!(result.is_ok(), "Flash Attention failed: {:?}", result.err());

        let output_buffer = result.unwrap();
        let output_len = n_heads * head_dim;

        // Read output
        let mut output = vec![0.0f32; output_len];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_buffer.contents() as *const f32,
                output.as_mut_ptr(),
                output_len,
            );
        }

        // Output should have reasonable values (not NaN or Inf)
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Output[{}] is not finite: {}", i, v);
        }
    }
}
