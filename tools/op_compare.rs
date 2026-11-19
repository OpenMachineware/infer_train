// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Operator comparison tool - compares our operators against llama.cpp
//
// Usage:
//   op_compare --op rms_norm --size 1024 --bench 1000
//   op_compare --op vec_dot_q5_k --verbose
//
// How it works:
//   1. Generate identical test data to a temporary file
//   2. Run op_test (Rust) and op_test_cpp (llama.cpp) with the same data
//   3. Compare output correctness and performance

use clap::Parser;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::process::Command;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "op_compare")]
#[command(
    about = "Operator comparison tool - compares our operators against llama.cpp"
)]
struct Cli {
    /// Operator name
    #[arg(short, long)]
    op: String,

    /// Input size
    #[arg(short, long, default_value = "1024")]
    size: usize,

    /// Number of benchmark iterations
    #[arg(long, default_value = "100")]
    bench: u32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Path to op_test binary
    #[arg(long, default_value = "target/release/op_test")]
    our_bin: String,

    /// Path to op_test_cpp binary
    #[arg(long, default_value = "tests/op_test/op_test_cpp")]
    llama_bin: String,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,
}

fn main() {
    let cli = Cli::parse();

    println!("=== Operator comparison: {} (size={}) ===\n", cli.op, cli.size);

    // Generate test data file
    let data_file = generate_test_data(&cli);
    println!("Test data: {}", data_file);

    // Build op_test first
    println!("\nBuilding op_test...");
    let build_status = Command::new("cargo")
        .args(["build", "--release", "--bin", "op_test"])
        .status()
        .expect("Failed to build op_test");

    if !build_status.success() {
        eprintln!("Failed to build op_test");
        std::process::exit(1);
    }

    // Run our implementation
    println!("\n--- Our implementation ---");
    let our_output = run_test(&cli.our_bin, &cli, &data_file, false);
    let our_bench = run_test(&cli.our_bin, &cli, &data_file, true);

    // Run llama.cpp implementation
    println!("\n--- llama.cpp implementation ---");
    let llama_output = run_test(&cli.llama_bin, &cli, &data_file, false);
    let llama_bench = run_test(&cli.llama_bin, &cli, &data_file, true);

    // Compare results
    println!("\n=== Comparison results ===");
    compare_outputs(&our_output, &llama_output, cli.verbose);
    compare_performance(&our_bench, &llama_bench);

    // Clean up temporary file
    std::fs::remove_file(&data_file).ok();
}

/// Generate test data file
fn generate_test_data(cli: &Cli) -> String {
    let filename = format!("/tmp/op_test_data_{}.bin", std::process::id());
    let file = File::create(&filename).expect("Failed to create data file");
    let mut writer = BufWriter::new(file);

    // Simple pseudo-random generator (LCG compatible with C++)
    let mut state = cli.seed;
    let mut gen_f32 = || -> f32 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((state >> 16) as i32 as f32) / 32768.0
    };

    // File format:
    // [header: op_name_len: u32, size: u32, seed: u64]
    // [op_name: bytes]
    // [input: size * f32]
    // [weight: size * f32] (for some operators)

    // Write header
    let op_bytes = cli.op.as_bytes();
    writer.write_all(&(op_bytes.len() as u32).to_le_bytes()).unwrap();
    writer.write_all(&(cli.size as u32).to_le_bytes()).unwrap();
    writer.write_all(&cli.seed.to_le_bytes()).unwrap();

    // Write op name
    writer.write_all(op_bytes).unwrap();

    // Write input
    for _ in 0..cli.size {
        let val = gen_f32();
        writer.write_all(&val.to_le_bytes()).unwrap();
    }

    // Write weight (for rms_norm etc)
    for _ in 0..cli.size {
        let val = gen_f32();
        writer.write_all(&val.to_le_bytes()).unwrap();
    }

    writer.flush().unwrap();
    filename
}

fn run_test(bin: &str, cli: &Cli, data_file: &str, bench_mode: bool) -> String {
    let start = Instant::now();

    let mut args = vec!["--data".to_string(), data_file.to_string()];

    if bench_mode {
        args.push("--bench".to_string());
        args.push(cli.bench.to_string());
    }

    if cli.verbose && !bench_mode {
        args.push("-v".to_string());
    }

    // Set library path for llama.cpp
    let llama_lib_path = std::path::PathBuf::from("llama.cpp-1.0.0/build/bin");

    let output = Command::new(bin)
        .args(&args)
        .env(
            "DYLD_LIBRARY_PATH",
            llama_lib_path
                .canonicalize()
                .unwrap_or_else(|_| llama_lib_path.clone()),
        )
        .output()
        .expect(&format!("Failed to run {}", bin));

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    if cli.verbose || bench_mode {
        println!("Command: {} {}", bin, args.join(" "));
        print!("{}", stdout);
    }

    if !output.status.success() {
        eprintln!("Execution failed: {} {}", bin, args.join(" "));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    }

    let elapsed = start.elapsed();
    if bench_mode {
        println!("Total time: {:?}", elapsed);
    }

    stdout
}

fn compare_outputs(our: &str, llama: &str, verbose: bool) {
    // Extract key values for comparison
    let our_values = extract_values(our);
    let llama_values = extract_values(llama);

    if our_values.is_empty() || llama_values.is_empty() {
        println!("Unable to extract output values for comparison");
        return;
    }

    if verbose {
        println!("Our output: {:?}", our_values);
        println!("llama.cpp output: {:?}", llama_values);
    }

    // Compare values
    let mut all_match = true;
    let tolerance = 1e-4;

    for (i, (a, b)) in our_values.iter().zip(llama_values.iter()).enumerate() {
        let diff = (a - b).abs();
        let rel_diff = if a.abs() > 1e-6 { diff / a.abs() } else { diff };

        if diff > tolerance && rel_diff > 0.01 {
            println!(
                "  [{}] Mismatch: ours={:.6}, llama={:.6}, diff={:.6}",
                i, a, b, diff
            );
            all_match = false;
        }
    }

    if all_match {
        println!("✓ Output matches (tolerance={})", tolerance);
    } else {
        println!("✗ Output mismatch");
    }
}

fn extract_values(output: &str) -> Vec<f64> {
    let mut values = Vec::new();

    // Extract "Result:" or similar values
    for line in output.lines() {
        // Look for "Result:" line
        if line.contains("Result:") || line.contains("result") {
            if let Some(num) = extract_number(line) {
                values.push(num);
            }
        }
        // Look for "Output [0..N]:" line
        if line.contains("Output [") || line.contains("Output:") {
            // Extract values in brackets
            for num in extract_numbers_from_brackets(line) {
                values.push(num);
            }
        }
        // Look for "Output range:" line
        if line.contains("Output range:") {
            for num in extract_numbers_from_brackets(line) {
                values.push(num);
            }
        }
    }

    values
}

fn extract_number(line: &str) -> Option<f64> {
    // Extract the last floating point number from the line
    let parts: Vec<&str> = line.split_whitespace().collect();
    for part in parts.iter().rev() {
        if let Ok(num) = part.trim_matches(',').parse::<f64>() {
            return Some(num);
        }
    }
    None
}

fn extract_numbers_from_brackets(line: &str) -> Vec<f64> {
    let mut numbers = Vec::new();

    // Find content between [ and ]
    if let Some(start) = line.find('[') {
        if let Some(end) = line.find(']') {
            let content = &line[start + 1..end];
            for part in content.split(',') {
                if let Ok(num) = part.trim().parse::<f64>() {
                    numbers.push(num);
                }
            }
        }
    }

    numbers
}

fn compare_performance(our_output: &str, llama_output: &str) {
    let our_time = extract_bench_time(our_output);
    let llama_time = extract_bench_time(llama_output);

    if let (Some(our), Some(llama)) = (our_time, llama_time) {
        let speedup = llama / our;
        println!("\nPerformance comparison:");
        println!("  Ours:    {:.2} µs/iter", our);
        println!("  llama:   {:.2} µs/iter", llama);

        if speedup > 1.0 {
            println!("  ✓ Speedup:  {:.2}x", speedup);
        } else {
            println!("  ✗ Slowdown:  {:.2}x", speedup);
        }

        // Target: every operator should be faster than llama.cpp
        if speedup >= 1.0 {
            println!("  Target:  Yes (≥100%)");
        } else {
            println!("  Target:  No (need ≥100%)");
        }
    } else {
        println!("Unable to extract performance data for comparison");
    }
}

fn extract_bench_time(output: &str) -> Option<f64> {
    // Look for "Per iteration:" line
    for line in output.lines() {
        if line.contains("Per iteration:") {
            // Extract µs value
            if let Some(idx) = line.find("µs") {
                let before = &line[..idx];
                if let Some(num) = extract_number(before) {
                    return Some(num);
                }
            }
            // May also be "us"
            if let Some(idx) = line.find("us") {
                let before = &line[..idx];
                if let Some(num) = extract_number(before) {
                    return Some(num);
                }
            }
        }
    }
    None
}
