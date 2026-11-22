// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Generic layer comparison tool
// Usage: layer_compare --model <path> <command> [options]
//
// Commands:
//   embedding           Compare embedding layer
//   layer <n>           Compare layer n output
//   logits              Compare final logits
//   full                Full layer-by-layer comparison
//   token <id>          Single token detailed trace
//   prefill             Prefill multiple tokens then compare logits
//
// Examples:
//   layer_compare -m model.gguf embedding -t 9707
//   layer_compare -m model.gguf token 9707
//   layer_compare -m model.gguf logits -p "Hello"
//   layer_compare -m model.gguf prefill -t 151644,8948,198,2610
//   layer_compare -m model.gguf full -t 9707,198 --trace-tokens 3
//   layer_compare -m model.gguf -v token 9707

use clap::{Parser, Subcommand};
use infer_train::inference::{ForwardExecutor, SimpleModel};
use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

#[derive(Parser)]
#[command(name = "layer_compare")]
#[command(about = "Generic layer comparison tool", version)]
struct Cli {
    /// GGUF model path
    #[arg(short, long)]
    model: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    mode: CompareMode,
}

#[derive(Subcommand)]
enum CompareMode {
    /// Compare embedding layer
    Embedding {
        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: Option<String>,
    },

    /// Compare specified layer output
    Layer {
        /// Layer number
        layer: usize,

        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: Option<String>,
    },

    /// Debug mode: output intermediate states for each layer
    Debug {
        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: Option<String>,

        /// Show only first N values per tensor
        #[arg(long, default_value = "5")]
        values: usize,

        /// Show only specific layer (0 = all)
        #[arg(long, default_value = "0")]
        layer: usize,
    },

    /// Compare final logits
    Logits {
        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: Option<String>,

        /// Show top N tokens
        #[arg(long, default_value = "10")]
        top_n: usize,
    },

    /// Full layer-by-layer comparison
    Full {
        /// Prompt text
        #[arg(short, long)]
        prompt: Option<String>,

        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: Option<String>,

        /// Only trace first N tokens
        #[arg(long, default_value = "3")]
        trace_tokens: usize,
    },

    /// Single token detailed trace
    Token {
        /// Token ID
        token_id: u32,
    },

    /// Prefill multiple tokens then compare final logits
    Prefill {
        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: String,

        /// Show top N tokens
        #[arg(long, default_value = "10")]
        top_n: usize,
    },

    /// Detailed trace with layer-by-layer output
    Trace {
        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: String,

        /// Show only first N layers (0 = all)
        #[arg(long, default_value = "0")]
        layers: usize,

        /// Show first N values per tensor
        #[arg(long, default_value = "5")]
        values: usize,
    },

    /// Prefill performance analysis
    Perf {
        /// Token IDs (comma separated)
        #[arg(short, long)]
        tokens: String,

        /// Number of iterations
        #[arg(long, default_value = "5")]
        iterations: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    let gguf = GgufFile::load(&cli.model).expect("Failed to load model");
    let meta =
        gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    let tokenizer = create_tokenizer(meta).expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(&cli.model, tokenizer).expect("Failed to load model");

    // Print model config
    let config = model.config();
    if cli.verbose {
        println!(
            "Model: {} layers, hidden={}, vocab={}",
            config.n_layers, config.hidden_size, config.vocab_size
        );
    }

    match cli.mode {
        CompareMode::Embedding { tokens } => {
            compare_embedding(&model, parse_tokens(tokens), cli.verbose);
        }
        CompareMode::Layer { layer, tokens } => {
            compare_layer(&model, layer, parse_tokens(tokens), cli.verbose);
        }
        CompareMode::Debug { tokens, values, layer } => {
            debug_layers(
                &cli.model,
                parse_tokens(tokens),
                values,
                layer,
                cli.verbose,
            );
        }
        CompareMode::Logits { prompt, tokens, top_n } => {
            compare_logits(
                &cli.model,
                parse_tokens_opt(&cli.model, prompt, tokens),
                top_n,
                cli.verbose,
            );
        }
        CompareMode::Full { prompt, tokens, trace_tokens } => {
            compare_full(
                &cli.model,
                parse_tokens_opt(&cli.model, prompt, tokens),
                trace_tokens,
                cli.verbose,
            );
        }
        CompareMode::Token { token_id } => {
            trace_single_token(&cli.model, token_id, cli.verbose);
        }
        CompareMode::Prefill { tokens, top_n } => {
            compare_prefill(
                &cli.model,
                parse_tokens(Some(tokens)),
                top_n,
                cli.verbose,
            );
        }
        CompareMode::Trace { tokens, layers, values } => {
            trace_detailed(
                &cli.model,
                parse_tokens(Some(tokens)),
                layers,
                values,
                cli.verbose,
            );
        }
        CompareMode::Perf { tokens, iterations } => {
            analyze_prefill_perf(
                &cli.model,
                parse_tokens(Some(tokens)),
                iterations,
            );
        }
    }
}

fn parse_tokens(tokens: Option<String>) -> Vec<u32> {
    match tokens {
        Some(s) => s.split(',').filter_map(|t| t.trim().parse().ok()).collect(),
        None => vec![9707], // Default "Hello"
    }
}

fn parse_tokens_opt(
    model_path: &str,
    prompt: Option<String>,
    tokens: Option<String>,
) -> Vec<u32> {
    if let Some(s) = tokens {
        return s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    }
    if let Some(p) = prompt {
        let gguf = GgufFile::load(model_path).unwrap();
        let meta = gguf.tokenizer_metadata().unwrap();
        let tokenizer = create_tokenizer(meta).unwrap();
        return tokenizer.encode(&p).unwrap();
    }
    vec![9707]
}

fn print_stats(name: &str, data: &[f32]) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
        / data.len() as f32)
        .sqrt();
    println!(
        "{}: shape=[{}], min={:.4}, max={:.4}, mean={:.4}, std={:.4}",
        name,
        data.len(),
        min,
        max,
        mean,
        std
    );
}

fn print_top_tokens(
    logits: &[f32],
    n: usize,
    tokenizer: &dyn infer_train::tokenizer::Tokenizer,
) {
    let mut indexed: Vec<(usize, f32)> =
        logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop {} tokens:", n);
    for (i, (token_id, logit)) in indexed.iter().take(n).enumerate() {
        let token_str = tokenizer
            .decode(&[*token_id as u32])
            .unwrap_or_else(|_| format!("token_{}", token_id));
        println!("  #{}: token {} = {:?}", i, token_id, token_str);
        println!("        logit = {:.4}", logit);
    }
}

fn compare_embedding(model: &SimpleModel, tokens: Vec<u32>, verbose: bool) {
    println!("\n=== Embedding Layer Comparison ===");
    println!("Tokens: {:?}", tokens);

    for token_id in tokens {
        let embedding =
            model.embed_token(token_id).expect("Failed to get embedding");
        print_stats(&format!("Embedding[{}]", token_id), &embedding);

        if verbose && embedding.len() >= 10 {
            println!("  First 10: {:?}", &embedding[..10]);
        }
    }
}

fn compare_layer(
    _model: &SimpleModel,
    layer: usize,
    tokens: Vec<u32>,
    _verbose: bool,
) {
    println!("\n=== Layer {} Comparison ===", layer);
    println!("Tokens: {:?}", tokens);
    println!("Note: Detailed layer comparison requires executor modification");
}

fn compare_logits(
    model_path: &str,
    tokens: Vec<u32>,
    top_n: usize,
    _verbose: bool,
) {
    println!("\n=== Logits Comparison ===");
    println!("Tokens: {:?}", tokens);

    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");

    // Show first token embedding
    if let Some(&first_token) = tokens.first() {
        let embedding =
            model.embed_token(first_token).expect("Failed to get embedding");
        print_stats(&format!("Embedding[{}]", first_token), &embedding);
    }

    let mut executor = ForwardExecutor::new(model);

    // Prefill all tokens
    for token in &tokens {
        let _ = executor.forward(*token).expect("Forward failed");
    }

    // Last token logits
    if let Some(&last_token) = tokens.last() {
        let logits =
            executor.forward(last_token).expect("Forward failed").to_vec();
        print_stats("Logits", &logits);
        print_top_tokens(&logits, top_n, executor.model.tokenizer());
    }
}

fn compare_full(
    model_path: &str,
    tokens: Vec<u32>,
    trace_tokens: usize,
    verbose: bool,
) {
    println!("\n=== Full Layer-by-Layer Comparison ===");
    println!("Tokens: {:?}", tokens);
    println!("Tracing first {} tokens", trace_tokens);

    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");
    let config = model.config();

    println!("\nModel config:");
    println!("  Layers: {}", config.n_layers);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Heads: {} (KV: {})", config.n_heads, config.n_kv_heads);
    println!("  Head dim: {}", config.head_dim());
    println!("  Vocab size: {}", config.vocab_size);

    let mut executor = ForwardExecutor::new(model);

    for (i, token) in tokens.iter().enumerate() {
        println!("\n=== Token {} (id={}) ===", i, token);

        // Show embedding (need to reload model)
        let gguf2 = GgufFile::load(model_path).expect("Failed to load model");
        let tok2 = create_tokenizer(
            gguf2.tokenizer_metadata().expect("Failed to get metadata"),
        )
        .expect("Failed to create tokenizer");
        let model2 =
            SimpleModel::load(model_path, tok2).expect("Failed to load model");
        let embedding =
            model2.embed_token(*token).expect("Failed to get embedding");
        print_stats("Embedding", &embedding);
        if verbose {
            println!("  First 5: {:?}", &embedding[..5.min(embedding.len())]);
        }

        let logits = executor.forward(*token).expect("Forward failed");

        if i < trace_tokens || i == tokens.len() - 1 {
            print_stats(&format!("Logits after token {}", i), &logits);

            let mut indexed: Vec<(usize, f32)> =
                logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!(
                "  Top 3: {:?}",
                indexed
                    .iter()
                    .take(3)
                    .map(|(id, logit)| (*id, *logit))
                    .collect::<Vec<_>>()
            );
        }
    }
}

fn trace_single_token(model_path: &str, token_id: u32, verbose: bool) {
    println!("\n=== Single Token Trace: {} ===", token_id);

    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");

    let embedding =
        model.embed_token(token_id).expect("Failed to get embedding");
    print_stats("Embedding", &embedding);

    if verbose {
        println!("\nFirst 20 embedding values:");
        for (i, v) in embedding.iter().take(20).enumerate() {
            println!("  [{}]: {:.6}", i, v);
        }
    }

    let gguf2 = GgufFile::load(model_path).expect("Failed to load model");
    let tok2 = create_tokenizer(
        gguf2.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model2 =
        SimpleModel::load(model_path, tok2).expect("Failed to load model");
    let mut executor = ForwardExecutor::new(model2);
    let logits = executor.forward(token_id).expect("Forward failed").to_vec();

    print_stats("Logits", &logits);
    print_top_tokens(&logits, 10, executor.model.tokenizer());
}

fn compare_prefill(
    model_path: &str,
    tokens: Vec<u32>,
    top_n: usize,
    verbose: bool,
) {
    println!("\n=== Prefill Comparison ===");
    println!("Tokens: {:?}", tokens);

    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");

    if verbose {
        let config = model.config();
        println!(
            "\nModel: {} layers, hidden={}, vocab={}",
            config.n_layers, config.hidden_size, config.vocab_size
        );
    }

    let mut executor = ForwardExecutor::new(model);

    // Prefill all tokens except the last one
    for token in &tokens[..tokens.len().saturating_sub(1)] {
        let _ = executor.forward(*token).expect("Forward failed");
    }

    // Process last token and get logits
    if let Some(&last_token) = tokens.last() {
        let logits =
            executor.forward(last_token).expect("Forward failed").to_vec();
        print_stats("Final Logits", &logits);
        print_top_tokens(&logits, top_n, executor.model.tokenizer());
    }
}

fn trace_detailed(
    model_path: &str,
    tokens: Vec<u32>,
    max_layers: usize,
    n_values: usize,
    verbose: bool,
) {
    println!("\n=== Detailed Layer Trace ===");
    println!("Tokens: {:?}", tokens);

    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");
    let config = model.config();

    if verbose {
        println!(
            "\nModel: {} layers, hidden={}, n_heads={}, n_kv_heads={}",
            config.n_layers,
            config.hidden_size,
            config.n_heads,
            config.n_kv_heads
        );

        // Print tensor info for first layer
        println!("\n=== First Layer Tensor Info ===");
        for name in &[
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ] {
            if let Some(tensor) = gguf.tensors.iter().find(|t| &t.name == name)
            {
                println!(
                    "{}: dims={:?}, strides={:?}, quant={:?}",
                    name, tensor.dims, tensor.strides, tensor.quant_type
                );
            }
        }

        // Print FFN weight tensor info
        println!("\n=== FFN Weight Tensor Info ===");
        for layer in 0..config.n_layers.min(4) {
            println!("\nLayer {}:", layer);
            for name in &[
                format!("blk.{}.ffn_gate.weight", layer),
                format!("blk.{}.ffn_up.weight", layer),
                format!("blk.{}.ffn_down.weight", layer),
            ] {
                if let Some(tensor) =
                    gguf.tensors.iter().find(|t| &t.name == name)
                {
                    println!(
                        "{}: dims={:?}, strides={:?}, quant={:?}",
                        name, tensor.dims, tensor.strides, tensor.quant_type
                    );
                    // Expected: dims=[in_features, out_features] = [1024, 3072] for gate/up
                    // strides[1] should be bytes per column
                }
            }
        }

        // Print FFN norm weight ranges
        println!("\n=== FFN Norm Weight Ranges ===");
        for layer in 0..config.n_layers.min(4) {
            let name = format!("blk.{}.ffn_norm.weight", layer);
            if let Some(data) = model.get_tensor(&name) {
                println!("{}: data_len={}", name, data.len());
                if data.len() == config.hidden_size * 4 {
                    let weights: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let min =
                        weights.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = weights
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    println!("  (F32) range: min={:.4}, max={:.4}", min, max);
                    // Print first few values
                    println!("  First 5: {:?}", &weights[..5]);
                } else if data.len() == config.hidden_size * 2 {
                    // F16 weights
                    let weights: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|c| {
                            half::f16::from_le_bytes([c[0], c[1]]).to_f32()
                        })
                        .collect();
                    let min =
                        weights.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = weights
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    println!("  (F16) range: min={:.4}, max={:.4}", min, max);
                    // Print first few values
                    println!("  First 5: {:?}", &weights[..5]);
                }
            } else {
                println!("{}: NOT FOUND", name);
            }
        }

        // Print Q/K norm weight ranges
        println!("\n=== Q/K Norm Weight Ranges ===");
        for name in &["blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight"] {
            if let Some(data) = model.get_tensor(name) {
                println!("{}: data_len={}", name, data.len());
                if data.len() == 512 {
                    // F32: 128 * 4
                    let weights: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let min =
                        weights.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = weights
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    println!("  (F32) min={:.4}, max={:.4}", min, max);
                } else if data.len() == 256 {
                    // F16: 128 * 2
                    let weights: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|c| {
                            half::f16::from_le_bytes([c[0], c[1]]).to_f32()
                        })
                        .collect();
                    let min =
                        weights.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = weights
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    println!("  (F16) min={:.4}, max={:.4}", min, max);
                }
            }
        }

        // Check last layer too
        println!("\n=== Last Layer Q/K Norm Weights ===");
        let last_layer = config.n_layers - 1;
        for name in &[
            format!("blk.{}.attn_q_norm.weight", last_layer),
            format!("blk.{}.attn_k_norm.weight", last_layer),
        ] {
            if let Some(data) = model.get_tensor(name) {
                if data.len() == 512 {
                    let weights: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let min =
                        weights.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = weights
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    println!("{}: min={:.4}, max={:.4}", name, min, max);
                }
            }
        }

        // Check all K norm weight ranges
        println!("\n=== All K Norm Weight Ranges ===");
        for layer in 0..config.n_layers {
            let name = format!("blk.{}.attn_k_norm.weight", layer);
            if let Some(data) = model.get_tensor(&name) {
                if data.len() == 512 {
                    let weights: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let min =
                        weights.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = weights
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    print!("L{:02}: [{:.2}, {:.2}]  ", layer, min, max);
                    if (layer + 1) % 4 == 0 {
                        println!();
                    }
                }
            }
        }
        println!();

        // Check output tensor
        println!("\n=== Output Projection ===");
        for name in &[
            "output.weight",
            "lm_head.weight",
            "output_norm.weight",
            "token_embd.weight",
        ] {
            if let Some(tensor) = gguf.tensors.iter().find(|t| &t.name == name)
            {
                println!(
                    "{}: dims={:?}, strides={:?}, quant={:?}",
                    name, tensor.dims, tensor.strides, tensor.quant_type
                );
            }
        }
    }

    let mut executor = ForwardExecutor::new(model);

    // Prefill all tokens except the last one
    println!("\n=== Prefill Phase ===");
    for (i, token) in
        tokens[..tokens.len().saturating_sub(1)].iter().enumerate()
    {
        let _ = executor.forward(*token).expect("Forward failed");
        println!(
            "Token {} (id={}): prefill complete, seq_pos={}",
            i, token, executor.seq_pos
        );
    }

    // Final forward to get logits
    if let Some(&last_token) = tokens.last() {
        let logits =
            executor.forward(last_token).expect("Forward failed").to_vec();
        println!("\n=== Final Output ===");
        print_stats("Logits", &logits);
        print_top_tokens(&logits, 10, executor.model.tokenizer());
    }
}

fn analyze_prefill_perf(model_path: &str, tokens: Vec<u32>, iterations: usize) {
    use std::time::Instant;

    println!("\n=== Prefill Performance Analysis ===");
    println!("Tokens: {} ({:?})", tokens.len(), tokens);
    println!("Iterations: {}", iterations);

    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");
    let config = model.config();

    println!(
        "\nModel config: {} layers, hidden={}, heads={}",
        config.n_layers, config.hidden_size, config.n_heads
    );

    let mut total_times = Vec::new();

    for iter in 0..iterations {
        // Reload model each iteration (can't clone)
        let gguf = GgufFile::load(model_path).expect("Failed to load model");
        let tokenizer = create_tokenizer(
            gguf.tokenizer_metadata().expect("Failed to get metadata"),
        )
        .expect("Failed to create tokenizer");
        let model = SimpleModel::load(model_path, tokenizer)
            .expect("Failed to load model");
        let mut executor = ForwardExecutor::new(model);

        let start = Instant::now();
        let _ = executor.forward_batch(&tokens).expect("Forward batch failed");
        let elapsed = start.elapsed();
        total_times.push(elapsed.as_millis() as f64);

        if iter == 0 {
            println!("\nFirst iteration time: {:.2} ms", elapsed.as_millis());
        }
    }

    let avg_ms: f64 = total_times.iter().sum::<f64>() / iterations as f64;
    let min_ms = total_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = total_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== Results ===");
    println!("Average: {:.2} ms", avg_ms);
    println!("Min: {:.2} ms", min_ms);
    println!("Max: {:.2} ms", max_ms);
    println!(
        "Throughput: {:.1} tokens/sec",
        tokens.len() as f64 / (avg_ms / 1000.0)
    );
}

fn debug_layers(
    model_path: &str,
    tokens: Vec<u32>,
    n_values: usize,
    target_layer: usize,
    verbose: bool,
) {
    use infer_train::inference::LayerDebugState;

    println!("\n=== Debug Mode: Layer-by-Layer Trace ===");
    println!("Tokens: {:?}", tokens);
    println!("Values per tensor: {}", n_values);
    if target_layer > 0 {
        println!("Target layer: {}", target_layer);
    }

    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");
    let config = model.config();

    if verbose {
        println!(
            "\nModel: {} layers, hidden={}, n_heads={}, n_kv_heads={}",
            config.n_layers,
            config.hidden_size,
            config.n_heads,
            config.n_kv_heads
        );
    }

    // Create executor with debug mode enabled
    let mut executor = ForwardExecutor::new(model);
    executor.debug_mode = true;

    // Process first token only (to see per-layer states)
    let token = tokens.first().copied().unwrap_or(9707);
    println!("\n=== Processing token {} ===", token);

    let _logits = executor.forward(token).expect("Forward failed");

    // Print debug states
    println!("\n=== Layer Debug States ===");
    for state in &executor.debug_states {
        if target_layer > 0 && state.layer_idx != target_layer {
            continue;
        }

        println!(
            "\n--- Layer {} (seq_pos={}) ---",
            state.layer_idx, state.seq_pos
        );
        print_debug_tensor("hidden_before", &state.hidden_before, n_values);
        print_debug_tensor("normed_attn", &state.normed_attn, n_values);
        print_debug_tensor("q", &state.q, n_values);
        print_debug_tensor("k", &state.k, n_values);
        print_debug_tensor("v", &state.v, n_values);
        print_debug_tensor("q_normed", &state.q_normed, n_values);
        print_debug_tensor("k_normed", &state.k_normed, n_values);
        print_debug_tensor("attn_output", &state.attn_output, n_values);
        print_debug_tensor("attn_out", &state.attn_out, n_values);
        // hidden_after_attn = hidden_before + attn_out (computed for display)
        let mut hidden_after_attn = state.hidden_before.clone();
        for i in 0..hidden_after_attn.len() {
            hidden_after_attn[i] += state.attn_out[i];
        }
        print_debug_tensor("hidden_after_attn", &hidden_after_attn, n_values);
        print_debug_tensor("normed_ffn", &state.normed_ffn, n_values);
        print_debug_tensor("  gate (before SiLU)", &state.gate, n_values);
        print_debug_tensor("  up", &state.up, n_values);
        print_debug_tensor(
            "  gate*up (SwiGLU)",
            &state.gate_times_up,
            n_values,
        );
        print_debug_tensor("ffn_out", &state.ffn_out, n_values);
        print_debug_tensor("hidden_after", &state.hidden_after, n_values);

        // Check for NaN/Inf
        if state.hidden_after.iter().any(|x| x.is_nan() || x.is_infinite()) {
            println!("\n*** WARNING: NaN/Inf detected in hidden_after! ***");
        }
    }

    // Print final logits
    println!("\n=== Final Logits ===");
    print_debug_tensor("logits", &executor.logits_buf, n_values);

    // Print top tokens
    let mut indexed: Vec<(usize, f32)> =
        executor.logits_buf.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 tokens:");
    for (i, (token_id, logit)) in indexed.iter().take(5).enumerate() {
        let token_str = executor
            .model
            .tokenizer()
            .decode(&[*token_id as u32])
            .unwrap_or_else(|_| format!("token_{}", token_id));
        println!(
            "  #{}: token {} = {:?} (logit={:.4})",
            i, token_id, token_str, logit
        );
    }
}

fn print_debug_tensor(name: &str, data: &[f32], n_values: usize) {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

    let has_nan = data.iter().any(|x| x.is_nan());
    let has_inf = data.iter().any(|x| x.is_infinite());

    let status = if has_nan {
        " [NaN!]"
    } else if has_inf {
        " [Inf!]"
    } else {
        ""
    };

    print!(
        "  {:<16}: shape={}, min={:+.4}, max={:+.4}, mean={:+.4}{}",
        name,
        data.len(),
        min,
        max,
        mean,
        status
    );

    if n_values > 0 {
        print!(" [");
        for (i, v) in data.iter().take(n_values).enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:+.4}", v);
        }
        if data.len() > n_values {
            print!(", ...");
        }
        print!("]");
    }
    println!();
}
