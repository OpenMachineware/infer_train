// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// KV Cache performance benchmark: memory, speed, accuracy

use criterion::{
    black_box, criterion_group, criterion_main, Criterion, Throughput,
};
use infer_train::inference::kv_cache::{KVCache, KVCacheBase, KVCacheFormat};
use infer_train::model::config::ModelConfig;
use std::collections::HashMap;

fn create_test_config() -> ModelConfig {
    ModelConfig {
        architecture: "llama".to_string(),
        hidden_size: 4096, // 128 head_dim with 32 heads
        intermediate_size: 11008,
        n_layers: 32,
        n_heads: 32,
        n_kv_heads: 32,
        vocab_size: 32000,
        rope_dim: 128,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 4096,
        bos_token_id: 1,
        eos_token_id: 2,
        metadata: HashMap::new(),
    }
}

/// Calculate theoretical memory usage
fn calc_memory_usage(
    format: KVCacheFormat,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
) -> usize {
    let total_elements = n_layers * n_kv_heads * max_seq * head_dim;

    match format {
        KVCacheFormat::F32 => total_elements * 4 * 2, // K + V
        KVCacheFormat::F16 => total_elements * 2 * 2,
        KVCacheFormat::BF16 => total_elements * 2 * 2,
        KVCacheFormat::Q8_0 => {
            let total_blocks = total_elements / 32;
            total_blocks * 34 * 2
        }
        KVCacheFormat::Q4_0 => {
            let total_blocks = total_elements / 32;
            total_blocks * 18 * 2
        }
        KVCacheFormat::Q4_1 => {
            let total_blocks = total_elements / 32;
            total_blocks * 20 * 2
        }
        KVCacheFormat::Q5_0 => {
            let total_blocks = total_elements / 32;
            total_blocks * 22 * 2
        }
        KVCacheFormat::Q5_1 => {
            let total_blocks = total_elements / 32;
            total_blocks * 24 * 2
        }
    }
}

fn bench_kv_cache(c: &mut Criterion) {
    let config = create_test_config();
    let max_seq = 1024;
    let head_dim = config.hidden_size / config.n_heads;
    let n_kv_heads = config.n_kv_heads;

    // Generate test data
    let k: Vec<f32> =
        (0..n_kv_heads * head_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let v: Vec<f32> =
        (0..n_kv_heads * head_dim).map(|i| (i as f32 * 0.01).cos()).collect();

    let formats = [
        (KVCacheFormat::F32, "F32"),
        (KVCacheFormat::F16, "F16"),
        (KVCacheFormat::Q8_0, "Q8_0"),
        (KVCacheFormat::Q4_0, "Q4_0"),
        (KVCacheFormat::Q4_1, "Q4_1"),
        (KVCacheFormat::Q5_0, "Q5_0"),
        (KVCacheFormat::Q5_1, "Q5_1"),
    ];

    // Memory usage comparison
    println!("\n=== KV Cache Memory Usage (32 layers, 1024 seq, 32 heads, 128 dim) ===");
    println!("{:8} {:>15} {:>10}", "Format", "Bytes", "Ratio");
    println!("{:-<8} {:-<15} {:-<10}", "", "", "");

    let f32_bytes = calc_memory_usage(
        KVCacheFormat::F32,
        config.n_layers,
        n_kv_heads,
        head_dim,
        max_seq,
    );

    for (format, name) in &formats {
        let bytes = calc_memory_usage(
            *format,
            config.n_layers,
            n_kv_heads,
            head_dim,
            max_seq,
        );
        let ratio = bytes as f64 / f32_bytes as f64;
        println!("{:8} {:>15} {:>9.1}%", name, bytes, ratio * 100.0);
    }

    // Speed benchmark: write 100 positions
    let mut group = c.benchmark_group("kv_cache_write");
    group.throughput(Throughput::Elements(100));

    for (format, name) in &formats {
        group.bench_function(*name, |b| {
            b.iter(|| {
                let mut cache = KVCache::new(&config, max_seq, *format);
                for _ in 0..100.min(max_seq) {
                    cache.set_k(0, black_box(&k), n_kv_heads, head_dim);
                    cache.set_v(0, black_box(&v), n_kv_heads, head_dim);
                    cache.advance();
                }
            })
        });
    }
    group.finish();

    // Speed benchmark: read 100 positions
    let mut group = c.benchmark_group("kv_cache_read");
    group.throughput(Throughput::Elements(100));

    for (format, name) in &formats {
        // Pre-fill cache
        let mut cache = KVCache::new(&config, max_seq, *format);
        for _ in 0..100.min(max_seq) {
            cache.set_k(0, &k, n_kv_heads, head_dim);
            cache.set_v(0, &v, n_kv_heads, head_dim);
            cache.advance();
        }

        group.bench_function(*name, |b| {
            b.iter(|| {
                for pos in 0..100.min(max_seq) {
                    let _ = cache.get_k(0, 0, black_box(pos));
                    let _ = cache.get_v(0, 0, black_box(pos));
                }
            })
        });
    }
    group.finish();

    // Accuracy comparison
    println!("\n=== KV Cache Accuracy (MSE vs F32) ===");
    println!("{:8} {:>12}", "Format", "MSE");
    println!("{:-<8} {:-<12}", "", "");

    let f32_cache = KVCache::new(&config, 2, KVCacheFormat::F32);
    let mut f32_cache = f32_cache;
    f32_cache.set_k(0, &k, n_kv_heads, head_dim);
    f32_cache.set_v(0, &v, n_kv_heads, head_dim);
    let f32_k = f32_cache.get_k(0, 0, 0);
    let f32_v = f32_cache.get_v(0, 0, 0);

    for (format, name) in &formats[1..] {
        let mut cache = KVCache::new(&config, 2, *format);
        cache.set_k(0, &k, n_kv_heads, head_dim);
        cache.set_v(0, &v, n_kv_heads, head_dim);
        let quant_k = cache.get_k(0, 0, 0);
        let quant_v = cache.get_v(0, 0, 0);

        let mse_k: f32 = f32_k
            .iter()
            .zip(quant_k.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let mse_v: f32 = f32_v
            .iter()
            .zip(quant_v.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let mse = (mse_k + mse_v) / (f32_k.len() + f32_v.len()) as f32;

        println!("{:8} {:>12.6}", name, mse);
    }
}

criterion_group!(benches, bench_kv_cache);
criterion_main!(benches);
