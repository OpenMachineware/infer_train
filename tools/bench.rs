// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Performance benchmark tool
//!
//! Usage:
//!   infer-train-bench -m <model.gguf> -p 512 -n 128
//!
//! Features:
//! - Prefill (pp) and decode (tg) benchmarking
//! - Multiple output formats: markdown, csv, json, jsonl, sql
//! - Statistical analysis with repetitions
//! - GPU and CPU device support
//! - Configurable batch size, threads, KV cache type

use clap::Parser;
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::model::GgufFile;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// LLM inference benchmark tool
#[derive(Parser, Debug)]
#[command(name = "infer-train-bench")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model path (GGUF file)
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Device to use (cpu or gpu)
    #[arg(short, long, default_value = "gpu")]
    device: String,

    /// Number of prompt tokens to process (prefill)
    /// Can specify multiple values separated by comma: -p 128,512,1024
    #[arg(short = 'p', long, default_value = "512")]
    n_prompt: String,

    /// Number of tokens to generate (decode)
    /// Can specify multiple values separated by comma: -n 128,256
    #[arg(short = 'n', long, default_value = "128")]
    n_gen: String,

    /// Combined prompt+gen pairs: -pg 512,128
    /// Can specify multiple: -pg 512,128 -pg 1024,256
    #[arg(long)]
    pg: Vec<String>,

    /// Batch size for prefill
    #[arg(short, long, default_value = "512")]
    batch_size: usize,

    /// Number of threads for CPU inference
    #[arg(short, long, default_value = "8")]
    threads: usize,

    /// Number of repetitions for each test
    #[arg(short, long, default_value = "5")]
    reps: usize,

    /// KV cache type (f32, f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl)
    #[arg(long, default_value = "f32")]
    kv_type: String,

    /// Output format (md, csv, json, jsonl, sql)
    #[arg(short = 'o', long, default_value = "md")]
    output: String,

    /// Output file (default: stdout)
    #[arg(short = 'O', long)]
    output_file: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip warmup
    #[arg(long)]
    no_warmup: bool,

    /// Delay between tests (seconds)
    #[arg(long, default_value = "0")]
    delay: u64,
}

/// Test configuration
#[derive(Debug, Clone)]
struct TestConfig {
    model: PathBuf,
    n_prompt: usize,
    n_gen: usize,
    device: Device,
    batch_size: usize,
    threads: usize,
    kv_type: KVCacheFormat,
}

/// Benchmark result
#[derive(Debug, Clone)]
struct TestResult {
    config: TestConfig,
    model_name: String,
    model_size: u64,
    model_params: u64,
    n_layers: usize,
    hidden_size: usize,
    n_heads: usize,
    test_time: String,
    samples_ns: Vec<u64>,
}

impl TestResult {
    fn avg_ns(&self) -> f64 {
        if self.samples_ns.is_empty() {
            return 0.0;
        }
        self.samples_ns.iter().sum::<u64>() as f64
            / self.samples_ns.len() as f64
    }

    fn stdev_ns(&self) -> f64 {
        if self.samples_ns.len() <= 1 {
            return 0.0;
        }
        let mean = self.avg_ns();
        let variance = self
            .samples_ns
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.samples_ns.len() - 1) as f64;
        variance.sqrt()
    }

    fn avg_ts(&self) -> f64 {
        let n_tokens = self.config.n_prompt + self.config.n_gen;
        if n_tokens == 0 {
            return 0.0;
        }
        let avg_ns = self.avg_ns();
        if avg_ns == 0.0 {
            return 0.0;
        }
        1e9 * n_tokens as f64 / avg_ns
    }

    fn stdev_ts(&self) -> f64 {
        let n_tokens = self.config.n_prompt + self.config.n_gen;
        if n_tokens == 0 || self.samples_ns.len() <= 1 {
            return 0.0;
        }
        let samples_ts: Vec<f64> = self
            .samples_ns
            .iter()
            .map(|&ns| 1e9 * n_tokens as f64 / ns as f64)
            .collect();
        let mean = samples_ts.iter().sum::<f64>() / samples_ts.len() as f64;
        let variance = samples_ts
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / (samples_ts.len() - 1) as f64;
        variance.sqrt()
    }

    fn test_name(&self) -> String {
        if self.config.n_prompt > 0 && self.config.n_gen == 0 {
            format!("pp{}", self.config.n_prompt)
        } else if self.config.n_gen > 0 && self.config.n_prompt == 0 {
            format!("tg{}", self.config.n_gen)
        } else {
            format!("pp{}+tg{}", self.config.n_prompt, self.config.n_gen)
        }
    }
}

fn parse_kv_type(s: &str) -> KVCacheFormat {
    match s.to_lowercase().as_str() {
        "f32" => KVCacheFormat::F32,
        "f16" => KVCacheFormat::F16,
        "bf16" => KVCacheFormat::BF16,
        "q8_0" => KVCacheFormat::Q8_0,
        "q4_0" => KVCacheFormat::Q4_0,
        "q4_1" => KVCacheFormat::Q4_1,
        "q5_0" => KVCacheFormat::Q5_0,
        "q5_1" => KVCacheFormat::Q5_1,
        _ => {
            eprintln!("Unknown KV type '{}', defaulting to f32", s);
            KVCacheFormat::F32
        }
    }
}

fn parse_int_list(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
}

fn parse_pg_list(list: &[String]) -> Vec<(usize, usize)> {
    list.iter()
        .filter_map(|s| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() == 2 {
                let p = parts[0].trim().parse().ok()?;
                let n = parts[1].trim().parse().ok()?;
                Some((p, n))
            } else {
                None
            }
        })
        .collect()
}

fn format_size(size: u64) -> String {
    if size < 1024 * 1024 * 1024 {
        format!("{:.2} MiB", size as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.2} GiB", size as f64 / 1024.0 / 1024.0 / 1024.0)
    }
}

fn format_params(params: u64) -> String {
    if params < 1_000_000_000 {
        format!("{:.2} M", params as f64 / 1e6)
    } else {
        format!("{:.2} B", params as f64 / 1e9)
    }
}

trait Printer {
    fn print_header(&mut self, configs: &[TestConfig]);
    fn print_result(&mut self, result: &TestResult);
    fn print_footer(&mut self);
}

struct MarkdownPrinter {
    out: BufWriter<Box<dyn Write>>,
    fields: Vec<String>,
}

impl MarkdownPrinter {
    fn new(out: BufWriter<Box<dyn Write>>) -> Self {
        Self { out, fields: Vec::new() }
    }

    fn field_width(field: &str) -> usize {
        match field {
            "model" => 30,
            "size" => 10,
            "params" => 10,
            "backend" => 8,
            "threads" => 7,
            "batch" => 7,
            "kv_type" => 6,
            "test" => 15,
            "t/s" => 20,
            _ => 10,
        }
    }

    fn display_name(field: &str) -> &str {
        match field {
            "n_prompt" | "n_gen" => "",
            "batch_size" => "batch",
            "threads" => "threads",
            "kv_type" => "kv",
            _ => field,
        }
    }
}

impl Printer for MarkdownPrinter {
    fn print_header(&mut self, configs: &[TestConfig]) {
        self.fields.clear();
        self.fields.push("model".to_string());
        self.fields.push("size".to_string());
        self.fields.push("params".to_string());
        self.fields.push("backend".to_string());

        // Only include fields that vary across tests
        let devices: Vec<_> = configs.iter().map(|c| c.device).collect();
        if devices.iter().any(|&d| d == Device::Cpu) {
            self.fields.push("threads".to_string());
        }

        self.fields.push("batch".to_string());
        self.fields.push("kv_type".to_string());
        self.fields.push("test".to_string());
        self.fields.push("t/s".to_string());

        write!(self.out, "|").unwrap_or(());
        for field in &self.fields {
            let width = Self::field_width(field);
            write!(self.out, " {:>width$} |", Self::display_name(field))
                .unwrap_or(());
        }
        writeln!(self.out).unwrap_or(());
        write!(self.out, "|").unwrap_or(());
        for field in &self.fields {
            let width = Self::field_width(field);
            write!(self.out, " {}{} |", "-".repeat(width - 1), ":")
                .unwrap_or(());
        }
        writeln!(self.out).unwrap_or(());
    }

    fn print_result(&mut self, result: &TestResult) {
        let values: std::collections::HashMap<&str, String> = [
            ("model", result.model_name.clone()),
            ("size", format_size(result.model_size)),
            ("params", format_params(result.model_params)),
            ("backend", format!("{:?}", result.config.device).to_lowercase()),
            ("threads", result.config.threads.to_string()),
            ("batch", result.config.batch_size.to_string()),
            ("kv_type", format!("{:?}", result.config.kv_type).to_lowercase()),
            ("test", result.test_name()),
            (
                "t/s",
                format!("{:.2} ± {:.2}", result.avg_ts(), result.stdev_ts()),
            ),
        ]
        .into_iter()
        .collect();

        write!(self.out, "|").unwrap_or(());
        for field in &self.fields {
            let width = Self::field_width(field);
            let value =
                values.get(field.as_str()).map(|s| s.as_str()).unwrap_or("");
            write!(self.out, " {:>width$} |", value).unwrap_or(());
        }
        writeln!(self.out).unwrap_or(());
    }

    fn print_footer(&mut self) {
        writeln!(self.out).unwrap_or(());
    }
}

struct CsvPrinter {
    out: BufWriter<Box<dyn Write>>,
}

impl CsvPrinter {
    fn new(out: BufWriter<Box<dyn Write>>) -> Self {
        Self { out }
    }

    fn escape_csv(s: &str) -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    }
}

impl Printer for CsvPrinter {
    fn print_header(&mut self, _configs: &[TestConfig]) {
        writeln!(
            self.out,
            "model,size,params,backend,threads,batch,kv_type,test,avg_ns,stddev_ns,avg_ts,stddev_ts"
        ).unwrap_or(());
    }

    fn print_result(&mut self, result: &TestResult) {
        writeln!(
            self.out,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            Self::escape_csv(&result.model_name),
            result.model_size,
            result.model_params,
            format!("{:?}", result.config.device).to_lowercase(),
            result.config.threads,
            result.config.batch_size,
            format!("{:?}", result.config.kv_type).to_lowercase(),
            result.test_name(),
            result.avg_ns() as u64,
            result.stdev_ns() as u64,
            format!("{:.2}", result.avg_ts()),
            format!("{:.2}", result.stdev_ts())
        )
        .unwrap_or(());
    }

    fn print_footer(&mut self) {}
}

struct JsonPrinter {
    out: BufWriter<Box<dyn Write>>,
    first: bool,
}

impl JsonPrinter {
    fn new(out: BufWriter<Box<dyn Write>>) -> Self {
        Self { out, first: true }
    }
}

impl Printer for JsonPrinter {
    fn print_header(&mut self, _configs: &[TestConfig]) {
        writeln!(self.out, "[").unwrap_or(());
    }

    fn print_result(&mut self, result: &TestResult) {
        if self.first {
            self.first = false;
        } else {
            writeln!(self.out, ",").unwrap_or(());
        }
        writeln!(self.out, "  {{").unwrap_or(());
        writeln!(self.out, "    \"model\": \"{}\",", result.model_name)
            .unwrap_or(());
        writeln!(self.out, "    \"size\": {},", result.model_size)
            .unwrap_or(());
        writeln!(self.out, "    \"params\": {},", result.model_params)
            .unwrap_or(());
        writeln!(
            self.out,
            "    \"backend\": \"{}\",",
            format!("{:?}", result.config.device).to_lowercase()
        )
        .unwrap_or(());
        writeln!(self.out, "    \"threads\": {},", result.config.threads)
            .unwrap_or(());
        writeln!(self.out, "    \"batch\": {},", result.config.batch_size)
            .unwrap_or(());
        writeln!(
            self.out,
            "    \"kv_type\": \"{}\",",
            format!("{:?}", result.config.kv_type).to_lowercase()
        )
        .unwrap_or(());
        writeln!(self.out, "    \"test\": \"{}\",", result.test_name())
            .unwrap_or(());
        writeln!(self.out, "    \"avg_ns\": {},", result.avg_ns() as u64)
            .unwrap_or(());
        writeln!(self.out, "    \"stddev_ns\": {},", result.stdev_ns() as u64)
            .unwrap_or(());
        writeln!(self.out, "    \"avg_ts\": {:.2},", result.avg_ts())
            .unwrap_or(());
        writeln!(self.out, "    \"stddev_ts\": {:.2},", result.stdev_ts())
            .unwrap_or(());
        writeln!(
            self.out,
            "    \"samples_ns\": [{}],",
            result
                .samples_ns
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap_or(());
        writeln!(
            self.out,
            "    \"samples_ts\": [{}]",
            result
                .samples_ns
                .iter()
                .map(|&ns| format!(
                    "{:.2}",
                    1e9 * (result.config.n_prompt + result.config.n_gen) as f64
                        / ns as f64
                ))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap_or(());
        write!(self.out, "  }}").unwrap_or(());
    }

    fn print_footer(&mut self) {
        writeln!(self.out).unwrap_or(());
        writeln!(self.out, "]").unwrap_or(());
    }
}

struct JsonlPrinter {
    out: BufWriter<Box<dyn Write>>,
}

impl JsonlPrinter {
    fn new(out: BufWriter<Box<dyn Write>>) -> Self {
        Self { out }
    }
}

impl Printer for JsonlPrinter {
    fn print_header(&mut self, _configs: &[TestConfig]) {}

    fn print_result(&mut self, result: &TestResult) {
        writeln!(
            self.out,
            "{{\"model\":\"{}\",\"size\":{},\"params\":{},\"backend\":\"{}\",\"threads\":{},\"batch\":{},\"kv_type\":\"{}\",\"test\":\"{}\",\"avg_ns\":{},\"stddev_ns\":{},\"avg_ts\":{:.2},\"stddev_ts\":{:.2},\"samples_ns\":[{}],\"samples_ts\":[{}]}}",
            result.model_name,
            result.model_size,
            result.model_params,
            format!("{:?}", result.config.device).to_lowercase(),
            result.config.threads,
            result.config.batch_size,
            format!("{:?}", result.config.kv_type).to_lowercase(),
            result.test_name(),
            result.avg_ns() as u64,
            result.stdev_ns() as u64,
            result.avg_ts(),
            result.stdev_ts(),
            result.samples_ns.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(","),
            result.samples_ns.iter()
                .map(|&ns| format!("{:.2}", 1e9 * (result.config.n_prompt + result.config.n_gen) as f64 / ns as f64))
                .collect::<Vec<_>>()
                .join(",")
        ).unwrap_or(());
    }

    fn print_footer(&mut self) {}
}

struct SqlPrinter {
    out: BufWriter<Box<dyn Write>>,
}

impl SqlPrinter {
    fn new(out: BufWriter<Box<dyn Write>>) -> Self {
        Self { out }
    }
}

impl Printer for SqlPrinter {
    fn print_header(&mut self, _configs: &[TestConfig]) {
        writeln!(self.out, "CREATE TABLE IF NOT EXISTS bench_results (")
            .unwrap_or(());
        writeln!(self.out, "  model TEXT,").unwrap_or(());
        writeln!(self.out, "  size INTEGER,").unwrap_or(());
        writeln!(self.out, "  params INTEGER,").unwrap_or(());
        writeln!(self.out, "  backend TEXT,").unwrap_or(());
        writeln!(self.out, "  threads INTEGER,").unwrap_or(());
        writeln!(self.out, "  batch INTEGER,").unwrap_or(());
        writeln!(self.out, "  kv_type TEXT,").unwrap_or(());
        writeln!(self.out, "  test TEXT,").unwrap_or(());
        writeln!(self.out, "  avg_ns INTEGER,").unwrap_or(());
        writeln!(self.out, "  stddev_ns INTEGER,").unwrap_or(());
        writeln!(self.out, "  avg_ts REAL,").unwrap_or(());
        writeln!(self.out, "  stddev_ts REAL").unwrap_or(());
        writeln!(self.out, ");").unwrap_or(());
        writeln!(self.out).unwrap_or(());
    }

    fn print_result(&mut self, result: &TestResult) {
        writeln!(
            self.out,
            "INSERT INTO bench_results VALUES ('{}', {}, {}, '{}', {}, {}, '{}', '{}', {}, {}, {:.2}, {:.2});",
            result.model_name,
            result.model_size,
            result.model_params,
            format!("{:?}", result.config.device).to_lowercase(),
            result.config.threads,
            result.config.batch_size,
            format!("{:?}", result.config.kv_type).to_lowercase(),
            result.test_name(),
            result.avg_ns() as u64,
            result.stdev_ns() as u64,
            result.avg_ts(),
            result.stdev_ts()
        ).unwrap_or(());
    }

    fn print_footer(&mut self) {}
}

fn create_printer(
    format: &str,
    out: BufWriter<Box<dyn Write>>,
) -> Box<dyn Printer> {
    match format.to_lowercase().as_str() {
        "md" | "markdown" => Box::new(MarkdownPrinter::new(out)),
        "csv" => Box::new(CsvPrinter::new(out)),
        "json" => Box::new(JsonPrinter::new(out)),
        "jsonl" => Box::new(JsonlPrinter::new(out)),
        "sql" => Box::new(SqlPrinter::new(out)),
        _ => {
            eprintln!(
                "Unknown output format '{}', defaulting to markdown",
                format
            );
            Box::new(MarkdownPrinter::new(out))
        }
    }
}

fn run_benchmark(
    config: &TestConfig,
    ctx: &mut InferenceContext,
    reps: usize,
    verbose: bool,
    no_warmup: bool,
) -> Option<Vec<u64>> {
    let n_prompt = config.n_prompt;
    let n_gen = config.n_gen;

    // Create KV cache
    let mut kv_cache = KVCache::new(
        &ctx.config,
        (n_prompt + n_gen + 256).max(512),
        config.kv_type,
    );

    // Generate random tokens for testing
    let vocab_size = ctx.config.vocab_size;
    let prompt_tokens: Vec<u32> =
        (0..n_prompt).map(|i| (i % (vocab_size - 1)) as u32 + 1).collect();

    // Warmup run
    if !no_warmup {
        if verbose {
            eprintln!("Warmup run...");
        }
        kv_cache.clear();
        if n_prompt > 0 {
            let _ = ctx.forward_pooled(&prompt_tokens, &mut kv_cache);
        }
        if n_gen > 0 {
            let token = if n_prompt > 0 { prompt_tokens[0] } else { 1 };
            let _ = ctx.forward_pooled(&[token], &mut kv_cache);
        }
    }

    let mut samples = Vec::with_capacity(reps);

    for rep in 0..reps {
        if verbose {
            eprintln!("Run {}/{}", rep + 1, reps);
        }

        kv_cache.clear();

        let start = Instant::now();

        // Prefill
        if n_prompt > 0 {
            let _ = ctx.forward_pooled(&prompt_tokens, &mut kv_cache);
        }

        // Decode
        let decode_token = if n_prompt > 0 { prompt_tokens[0] } else { 1 };
        for _ in 0..n_gen {
            let _ = ctx.forward_pooled(&[decode_token], &mut kv_cache);
        }

        let elapsed = start.elapsed().as_nanos() as u64;
        samples.push(elapsed);
    }

    Some(samples)
}

fn main() {
    let args = Args::parse();

    // Parse device
    let device = match args.device.to_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "gpu" => Device::Gpu,
        _ => {
            eprintln!("Invalid device '{}', using GPU", args.device);
            Device::Gpu
        }
    };

    // Parse test configurations
    let n_prompts = parse_int_list(&args.n_prompt);
    let n_gens = parse_int_list(&args.n_gen);
    let pg_pairs = parse_pg_list(&args.pg);
    let kv_type = parse_kv_type(&args.kv_type);

    // Build test configs
    let mut configs: Vec<TestConfig> = Vec::new();

    // Add prompt-only tests
    for &n_prompt in &n_prompts {
        if n_prompt > 0 {
            configs.push(TestConfig {
                model: args.model.clone(),
                n_prompt,
                n_gen: 0,
                device,
                batch_size: args.batch_size,
                threads: args.threads,
                kv_type,
            });
        }
    }

    // Add generation-only tests
    for &n_gen in &n_gens {
        if n_gen > 0 {
            configs.push(TestConfig {
                model: args.model.clone(),
                n_prompt: 0,
                n_gen,
                device,
                batch_size: args.batch_size,
                threads: args.threads,
                kv_type,
            });
        }
    }

    // Add combined tests
    for &(n_prompt, n_gen) in &pg_pairs {
        configs.push(TestConfig {
            model: args.model.clone(),
            n_prompt,
            n_gen,
            device,
            batch_size: args.batch_size,
            threads: args.threads,
            kv_type,
        });
    }

    // If no configs, add default
    if configs.is_empty() {
        configs.push(TestConfig {
            model: args.model.clone(),
            n_prompt: 512,
            n_gen: 128,
            device,
            batch_size: args.batch_size,
            threads: args.threads,
            kv_type,
        });
    }

    // Load model
    if args.verbose {
        eprintln!("Loading model from: {}", args.model.display());
    }

    let mut ctx =
        match InferenceContext::from_gguf(args.model.to_str().unwrap(), device)
        {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                std::process::exit(1);
            }
        };

    // Get model info
    let gguf = match GgufFile::open(args.model.to_str().unwrap()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open GGUF: {}", e);
            std::process::exit(1);
        }
    };

    let model_name =
        args.model.file_name().unwrap().to_str().unwrap().to_string();
    let model_size =
        std::fs::metadata(&args.model).map(|m| m.len()).unwrap_or(0);
    let model_params: u64 = (ctx.config.hidden_size * ctx.config.vocab_size
        / 1000000
        * ctx.config.n_layers as usize) as u64;

    if args.verbose {
        eprintln!(
            "Model: {} layers, {} hidden, {} heads",
            ctx.config.n_layers, ctx.config.hidden_size, ctx.config.n_heads
        );
        eprintln!("Device: {:?}", device);
        eprintln!("Batch size: {}", args.batch_size);
        eprintln!("Threads: {}", args.threads);
        eprintln!("KV cache type: {:?}", kv_type);
        eprintln!();
    }

    // Setup output
    let out: BufWriter<Box<dyn Write>> = match &args.output_file {
        Some(path) => BufWriter::new(Box::new(
            File::create(path).expect("Failed to create output file"),
        )),
        None => BufWriter::new(Box::new(std::io::stdout())),
    };

    let mut printer = create_printer(&args.output, out);
    printer.print_header(&configs);

    // Run benchmarks
    for (i, config) in configs.iter().enumerate() {
        if args.verbose {
            eprintln!(
                "Test {}/{}: {}",
                i + 1,
                configs.len(),
                TestResult {
                    config: config.clone(),
                    model_name: model_name.clone(),
                    model_size,
                    model_params,
                    n_layers: ctx.config.n_layers as usize,
                    hidden_size: ctx.config.hidden_size,
                    n_heads: ctx.config.n_heads,
                    test_time: String::new(),
                    samples_ns: vec![],
                }
                .test_name()
            );
        }

        let samples = run_benchmark(
            config,
            &mut ctx,
            args.reps,
            args.verbose,
            args.no_warmup,
        );

        if let Some(samples) = samples {
            let test_time =
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

            let result = TestResult {
                config: config.clone(),
                model_name: model_name.clone(),
                model_size,
                model_params,
                n_layers: ctx.config.n_layers as usize,
                hidden_size: ctx.config.hidden_size,
                n_heads: ctx.config.n_heads,
                test_time: test_time.to_string(),
                samples_ns: samples,
            };

            printer.print_result(&result);
        }

        // Delay between tests
        if args.delay > 0 {
            std::thread::sleep(std::time::Duration::from_secs(args.delay));
        }
    }

    printer.print_footer();
}
