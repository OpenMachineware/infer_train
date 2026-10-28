// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Benchmark CPU DAG Executor vs regular forward

use infer_train::dag::{CpuDagExecutor, DagBuilder};
use infer_train::inference::{KVCache, KVCacheFormat};
use infer_train::model::weights::WeightStore;
use infer_train::model::{GgufFile, LlamaModel, ModelConfig};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("Loading model from {}...", model_path);

    // Load model
    let store =
        WeightStore::from_gguf(model_path).expect("Failed to load weights");
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let raw_config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, raw_config);

    let hidden_size = model.config.hidden_size;
    let n_layers = model.layers.len();
    let n_heads = model.config.n_heads;
    let n_kv_heads = model.config.n_kv_heads;
    let head_dim = model.config.head_dim;
    let vocab_size = model.config.vocab_size;

    println!("Model config:");
    println!("  hidden_size: {}", hidden_size);
    println!("  n_layers: {}", n_layers);
    println!("  n_heads: {}", n_heads);
    println!("  n_kv_heads: {}", n_kv_heads);
    println!("  head_dim: {}", head_dim);
    println!("  vocab_size: {}", vocab_size);

    // Build DAG
    println!("\nBuilding DAG...");
    let dag = DagBuilder::new().build_model_dag(&model);
    let optimized_dag = dag.into_optimized();

    println!("DAG nodes: {}", optimized_dag.node_count());
    println!("DAG edges: {}", optimized_dag.edge_count());

    // Test tokens
    let tokens = vec![1u32]; // Single token for faster testing

    // Initialize CPU DAG executor
    let mut cpu_dag_executor = CpuDagExecutor::new();

    // Warmup
    println!("\nWarmup...");
    for i in 0..3 {
        println!("  Warmup iteration {}...", i);
        let mut kv_cache =
            KVCache::new(&model.config, 2048, KVCacheFormat::F32);
        let result = model.forward(&tokens, &mut kv_cache);
        println!("    Result: {:?}", result.as_ref().map(|r| r.len()));
    }

    // Benchmark regular forward
    println!("\nBenchmarking regular forward...");
    let n_iterations = 3; // Reduced for testing
    let mut total_time = 0u128;

    for i in 0..n_iterations {
        let mut kv_cache =
            KVCache::new(&model.config, 2048, KVCacheFormat::F32);
        let start = Instant::now();
        let result = model.forward(&tokens, &mut kv_cache);
        let elapsed = start.elapsed().as_nanos();
        total_time += elapsed;

        if i == 0 {
            if let Some(logits) = result {
                println!("Regular forward output size: {}", logits.len());
            }
        }
    }

    let avg_regular = total_time / n_iterations as u128;
    println!("Regular forward avg: {:.2} ms", avg_regular as f64 / 1_000_000.0);

    // Benchmark CPU DAG executor
    println!("\nBenchmarking CPU DAG executor...");
    let mut total_time = 0u128;

    for i in 0..n_iterations {
        let mut kv_cache =
            KVCache::new(&model.config, 2048, KVCacheFormat::F32);
        let start = Instant::now();
        let result = cpu_dag_executor.forward(
            &model,
            &tokens,
            &mut kv_cache,
            &optimized_dag,
            &optimized_dag.dag.weight_bindings,
        );
        let elapsed = start.elapsed().as_nanos();
        total_time += elapsed;

        if i == 0 {
            if let Some(logits) = result {
                println!("CPU DAG executor output size: {}", logits.len());
            } else {
                println!("CPU DAG executor returned None!");
            }
        }
    }

    let avg_dag = total_time / n_iterations as u128;
    println!("CPU DAG executor avg: {:.2} ms", avg_dag as f64 / 1_000_000.0);

    // Compare
    println!("\n=== Comparison ===");
    println!("Regular forward: {:.2} ms", avg_regular as f64 / 1_000_000.0);
    println!("CPU DAG executor: {:.2} ms", avg_dag as f64 / 1_000_000.0);
    if avg_dag > 0 {
        let ratio = avg_regular as f64 / avg_dag as f64;
        if ratio > 1.0 {
            println!("CPU DAG is {:.2}x faster", ratio);
        } else {
            println!("CPU DAG is {:.2}x slower", 1.0 / ratio);
        }
    }
}
