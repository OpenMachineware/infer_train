// Benchmark thread creation overhead

use std::thread;
use std::time::Instant;

fn main() {
    let n_threads = 10;
    let n_iterations = 1000;

    println!("=== Thread Creation Overhead Benchmark ===");
    println!("n_threads: {}, iterations: {}\n", n_threads, n_iterations);

    // Benchmark 1: Thread scope creation overhead
    let mut total_time = 0u64;
    for _ in 0..n_iterations {
        let start = Instant::now();
        thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    // Do nothing
                });
            }
        });
        total_time += start.elapsed().as_nanos() as u64;
    }

    println!("Thread::scope overhead (empty threads):");
    println!("  Total: {:.2} ms", total_time as f64 / 1_000_000.0);
    println!(
        "  Per iteration: {:.2} µs",
        total_time as f64 / n_iterations as f64 / 1000.0
    );
    println!(
        "  Per thread: {:.2} µs",
        total_time as f64 / (n_iterations * n_threads) as f64 / 1000.0
    );

    // Benchmark 2: Compare with single-threaded
    let mut single_time = 0u64;
    for _ in 0..n_iterations {
        let start = Instant::now();
        for _ in 0..n_threads {
            // Do nothing
        }
        single_time += start.elapsed().as_nanos() as u64;
    }

    println!("\nSingle-threaded loop (baseline):");
    println!("  Total: {:.2} ms", single_time as f64 / 1_000_000.0);
    println!(
        "  Per iteration: {:.2} µs",
        single_time as f64 / n_iterations as f64 / 1000.0
    );

    let overhead =
        (total_time - single_time) as f64 / n_iterations as f64 / 1000.0;
    println!("\nThread creation overhead per iteration: {:.2} µs", overhead);

    // Estimate for typical forward pass
    let n_gemv_per_layer = 7; // wq, wk, wv, wo, w1, w2, w3
    let n_layers = 32;
    let total_gemv = n_gemv_per_layer * n_layers;

    println!("\n=== Estimated overhead per forward pass ===");
    println!(
        "GEMV calls: {} ({} layers × {} GEMV)",
        total_gemv, n_layers, n_gemv_per_layer
    );
    println!(
        "Thread creation overhead: {:.2} ms",
        overhead * total_gemv as f64 / 1000.0
    );
}
