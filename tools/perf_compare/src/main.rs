// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Performance comparison tool: infer_train vs llama.cpp
//!
//! Measures:
//! - Prefill throughput (tokens/sec)
//! - Decode throughput (tokens/sec)
//! - Memory usage
//! - Thread count

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "perf_compare")]
#[command(about = "Compare inference performance with llama.cpp")]
struct Cli {
    /// Model path
    #[arg(short, long)]
    model: String,

    /// Prompt for testing
    #[arg(short, long, default_value = "Hello, how are you today?")]
    prompt: String,

    /// Number of decode tokens
    #[arg(long, default_value = "32")]
    decode_tokens: usize,

    /// Number of warmup iterations
    #[arg(long, default_value = "2")]
    warmup: usize,

    /// Number of measurement iterations
    #[arg(long, default_value = "3")]
    iterations: usize,

    /// Path to llama-cli binary
    #[arg(long, default_value = "llama-cli")]
    llama_cli: String,

    /// Output JSON report
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerfMetrics {
    name: String,
    prefill_time_ms: f64,
    prefill_tokens: usize,
    prefill_tps: f64,
    decode_time_ms: f64,
    decode_tokens: usize,
    decode_tps: f64,
    total_time_ms: f64,
    total_tps: f64,
    memory_mb: f64,
}

impl PerfMetrics {
    fn ratio(&self, other: &PerfMetrics) -> PerfRatio {
        PerfRatio {
            prefill_ratio: self.prefill_tps / other.prefill_tps,
            decode_ratio: self.decode_tps / other.decode_tps,
            total_ratio: self.total_tps / other.total_tps,
        }
    }
}

#[derive(Debug, Serialize)]
struct PerfRatio {
    prefill_ratio: f64,
    decode_ratio: f64,
    total_ratio: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonResult {
    ours: PerfMetrics,
    llama_cpp: PerfMetrics,
    ratio: PerfRatio,
}

fn main() {
    let cli = Cli::parse();

    println!("=== Performance Comparison ===");
    println!("Model: {}", cli.model);
    println!("Prompt: {}", cli.prompt);
    println!("Decode tokens: {}", cli.decode_tokens);
    println!("Warmup: {}, Iterations: {}", cli.warmup, cli.iterations);
    println!();

    // Warmup
    println!("=== Warmup ===");
    for i in 0..cli.warmup {
        println!("Warmup {}/{} (ours)...", i + 1, cli.warmup);
        let _ = run_ours(&cli.model, &cli.prompt, 10);
    }
    for i in 0..cli.warmup {
        println!("Warmup {}/{} (llama.cpp)...", i + 1, cli.warmup);
        let _ = run_llama_cpp(&cli.llama_cli, &cli.model, &cli.prompt, 10);
    }
    println!();

    // Measure
    println!("=== Measurement ===");
    let mut ours_results = Vec::new();
    let mut llama_results = Vec::new();

    for i in 0..cli.iterations {
        println!("Iteration {}/{}", i + 1, cli.iterations);

        // Run ours
        let ours = run_ours(&cli.model, &cli.prompt, cli.decode_tokens);
        ours_results.push(ours);

        // Run llama.cpp
        let llama = run_llama_cpp(
            &cli.llama_cli,
            &cli.model,
            &cli.prompt,
            cli.decode_tokens,
        );
        llama_results.push(llama);
    }

    // Average results
    let ours_avg = average_metrics(&ours_results, "infer_train");
    let llama_avg = average_metrics(&llama_results, "llama.cpp");

    // Calculate ratio
    let ratio = ours_avg.ratio(&llama_avg);

    // Output
    if cli.json {
        let result = ComparisonResult {
            ours: ours_avg.clone(),
            llama_cpp: llama_avg.clone(),
            ratio,
        };
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        print_comparison_table(&ours_avg, &llama_avg, &ratio);
    }
}

fn run_ours(
    model_path: &str,
    prompt: &str,
    decode_tokens: usize,
) -> PerfMetrics {
    use infer_train::inference::{ForwardExecutor, SimpleModel};
    use infer_train::model::GgufFile;
    use infer_train::tokenizer::create_tokenizer;

    let gguf = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(model_path, tokenizer).expect("Failed to load model");

    // Re-create tokenizer for encoding
    let gguf2 = GgufFile::load(model_path).expect("Failed to load model");
    let tokenizer2 = create_tokenizer(
        gguf2.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");

    let mut executor = ForwardExecutor::new(model);

    // Tokenize prompt
    let tokens = tokenizer2.encode(prompt).expect("Failed to tokenize");
    let prefill_tokens = tokens.len();

    // Get memory before
    let memory_before = get_memory_mb();

    // Prefill
    let prefill_start = Instant::now();
    for &token in &tokens {
        executor.forward(token).expect("Forward failed");
    }
    let prefill_time = prefill_start.elapsed();

    // Decode
    let decode_start = Instant::now();
    let mut generated = 0usize;
    let mut last_token = *tokens.last().unwrap();

    for _ in 0..decode_tokens {
        let logits = executor.forward(last_token).expect("Forward failed");
        // Simple greedy sampling
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        last_token = max_idx as u32;
        generated += 1;
    }
    let decode_time = decode_start.elapsed();

    // Get memory after
    let memory_after = get_memory_mb();

    PerfMetrics {
        name: "infer_train".to_string(),
        prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
        prefill_tokens,
        prefill_tps: prefill_tokens as f64 / prefill_time.as_secs_f64(),
        decode_time_ms: decode_time.as_secs_f64() * 1000.0,
        decode_tokens: generated,
        decode_tps: generated as f64 / decode_time.as_secs_f64(),
        total_time_ms: (prefill_time + decode_time).as_secs_f64() * 1000.0,
        total_tps: (prefill_tokens + generated) as f64
            / (prefill_time + decode_time).as_secs_f64(),
        memory_mb: memory_after - memory_before,
    }
}

fn run_llama_cpp(
    llama_cli: &str,
    model_path: &str,
    prompt: &str,
    decode_tokens: usize,
) -> PerfMetrics {
    // Build command
    let output = Command::new(llama_cli)
        .arg("-m")
        .arg(model_path)
        .arg("-p")
        .arg(prompt)
        .arg("-n")
        .arg(decode_tokens.to_string())
        .arg("--temp")
        .arg("0")
        .arg("--no-display-prompt")
        .output()
        .expect("Failed to run llama-cli");

    let _stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Parse llama.cpp output for metrics
    // llama.cpp outputs something like:
    // "prefill time = 123.45 ms / 10 tokens"
    // "decode time = 234.56 ms / 32 tokens"

    let mut prefill_time_ms = 0.0f64;
    let mut prefill_tokens = 0usize;
    let mut decode_time_ms = 0.0f64;
    let mut decode_tokens = 0usize;

    // Parse from stderr (llama.cpp logs there)
    for line in stderr.lines() {
        // Look for timing info
        if line.contains("eval time") || line.contains("prompt eval time") {
            // Example: "prompt eval time = 123.45 ms / 10 tokens (81.23 tokens per second)"
            if let Some(time_str) = extract_number_before(line, " ms") {
                prefill_time_ms = time_str;
            }
            if let Some(tokens_str) =
                extract_number_after(line, "/ ", " tokens")
            {
                prefill_tokens = tokens_str;
            }
        }
        if line.contains("generation eval time") || line.contains("eval time") {
            if line.contains("tokens per second") {
                if let Some(time_str) = extract_number_before(line, " ms") {
                    if decode_time_ms == 0.0 {
                        decode_time_ms = time_str;
                    }
                }
                if let Some(tokens_str) =
                    extract_number_after(line, "/ ", " tokens")
                {
                    if decode_tokens == 0 {
                        decode_tokens = tokens_str;
                    }
                }
            }
        }
    }

    // Fallback: estimate from stdout token count
    if prefill_tokens == 0 {
        prefill_tokens = prompt.split_whitespace().count();
    }

    let memory_mb = 0.0; // Would need ps to get memory

    PerfMetrics {
        name: "llama.cpp".to_string(),
        prefill_time_ms,
        prefill_tokens,
        prefill_tps: if prefill_time_ms > 0.0 {
            prefill_tokens as f64 / (prefill_time_ms / 1000.0)
        } else {
            0.0
        },
        decode_time_ms,
        decode_tokens,
        decode_tps: if decode_time_ms > 0.0 {
            decode_tokens as f64 / (decode_time_ms / 1000.0)
        } else {
            0.0
        },
        total_time_ms: prefill_time_ms + decode_time_ms,
        total_tps: if prefill_time_ms + decode_time_ms > 0.0 {
            (prefill_tokens + decode_tokens) as f64
                / ((prefill_time_ms + decode_time_ms) / 1000.0)
        } else {
            0.0
        },
        memory_mb,
    }
}

fn extract_number_before(s: &str, suffix: &str) -> Option<f64> {
    if let Some(pos) = s.find(suffix) {
        let before = &s[..pos];
        // Find the last number in before
        for word in before.split_whitespace().rev() {
            if let Ok(n) = word.parse::<f64>() {
                return Some(n);
            }
        }
    }
    None
}

fn extract_number_after(s: &str, prefix: &str, suffix: &str) -> Option<usize> {
    if let Some(start) = s.find(prefix) {
        let after = &s[start + prefix.len()..];
        if let Some(end) = after.find(suffix) {
            let num_str = after[..end].trim();
            if let Ok(n) = num_str.parse::<usize>() {
                return Some(n);
            }
        }
    }
    None
}

fn get_memory_mb() -> f64 {
    // Get current process memory on macOS/Linux
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output();
        if let Ok(out) = output {
            let rss_kb: f64 = String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse()
                .unwrap_or(0.0);
            return rss_kb / 1024.0; // Convert to MB
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
    }
    0.0
}

fn average_metrics(metrics: &[PerfMetrics], name: &str) -> PerfMetrics {
    let n = metrics.len() as f64;
    PerfMetrics {
        name: name.to_string(),
        prefill_time_ms: metrics.iter().map(|m| m.prefill_time_ms).sum::<f64>()
            / n,
        prefill_tokens: metrics[0].prefill_tokens,
        prefill_tps: metrics.iter().map(|m| m.prefill_tps).sum::<f64>() / n,
        decode_time_ms: metrics.iter().map(|m| m.decode_time_ms).sum::<f64>()
            / n,
        decode_tokens: metrics[0].decode_tokens,
        decode_tps: metrics.iter().map(|m| m.decode_tps).sum::<f64>() / n,
        total_time_ms: metrics.iter().map(|m| m.total_time_ms).sum::<f64>() / n,
        total_tps: metrics.iter().map(|m| m.total_tps).sum::<f64>() / n,
        memory_mb: metrics.iter().map(|m| m.memory_mb).sum::<f64>() / n,
    }
}

fn print_comparison_table(
    ours: &PerfMetrics,
    llama: &PerfMetrics,
    ratio: &PerfRatio,
) {
    println!("\n=== Results ===\n");

    println!(
        "{:20} {:>15} {:>15} {:>10}",
        "Metric", "infer_train", "llama.cpp", "Ratio"
    );
    println!("{}", "-".repeat(65));

    println!(
        "{:20} {:>12.2} ms {:>12.2} ms {:>10}",
        "Prefill time", ours.prefill_time_ms, llama.prefill_time_ms, ""
    );

    println!(
        "{:20} {:>12.1} t/s {:>12.1} t/s {:>9.2}x",
        "Prefill throughput",
        ours.prefill_tps,
        llama.prefill_tps,
        ratio.prefill_ratio
    );

    println!(
        "{:20} {:>12.2} ms {:>12.2} ms {:>10}",
        "Decode time", ours.decode_time_ms, llama.decode_time_ms, ""
    );

    println!(
        "{:20} {:>12.1} t/s {:>12.1} t/s {:>9.2}x",
        "Decode throughput",
        ours.decode_tps,
        llama.decode_tps,
        ratio.decode_ratio
    );

    println!(
        "{:20} {:>12.2} ms {:>12.2} ms {:>10}",
        "Total time", ours.total_time_ms, llama.total_time_ms, ""
    );

    println!(
        "{:20} {:>12.1} t/s {:>12.1} t/s {:>9.2}x",
        "Total throughput", ours.total_tps, llama.total_tps, ratio.total_ratio
    );

    println!();

    // Summary
    println!("=== Summary ===");
    if ratio.decode_ratio > 1.0 {
        println!("Decode: {:.2}x faster than llama.cpp", ratio.decode_ratio);
    } else {
        println!(
            "Decode: {:.2}x slower than llama.cpp",
            1.0 / ratio.decode_ratio
        );
    }

    if ratio.prefill_ratio > 1.0 {
        println!("Prefill: {:.2}x faster than llama.cpp", ratio.prefill_ratio);
    } else {
        println!(
            "Prefill: {:.2}x slower than llama.cpp",
            1.0 / ratio.prefill_ratio
        );
    }
}
