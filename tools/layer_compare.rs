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
        let logits = executor.forward(last_token).expect("Forward failed");
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
    let logits = executor.forward(token_id).expect("Forward failed");

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

    // Prefill all tokens
    for token in &tokens {
        let _ = executor.forward(*token).expect("Forward failed");
    }

    // Last token logits
    if let Some(&last_token) = tokens.last() {
        let logits = executor.forward(last_token).expect("Forward failed");
        print_stats("Final Logits", &logits);
        print_top_tokens(&logits, top_n, executor.model.tokenizer());
    }
}
