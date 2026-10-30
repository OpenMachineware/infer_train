// Benchmark thread creation overhead for GEMV operations

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn main() {
    let n_threads =
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    let n_iterations = 1000;

    println!("=== Thread Overhead Benchmark ===");
    println!("Threads: {}", n_threads);

    // Test 1: Create threads each time (current approach)
    let start = Instant::now();
    for _ in 0..n_iterations {
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    // Simulate small work
                    let mut sum = 0u64;
                    for i in 0..100 {
                        sum += i;
                    }
                    std::hint::black_box(sum);
                });
            }
        });
    }
    let create_each_time = start.elapsed();
    println!(
        "Create threads each time: {:?} ({:.2} µs/iter)",
        create_each_time,
        create_each_time.as_micros() as f64 / n_iterations as f64
    );

    // Test 2: Single thread (no overhead)
    let start = Instant::now();
    for _ in 0..n_iterations {
        let mut sum = 0u64;
        for i in 0..100 {
            sum += i;
        }
        std::hint::black_box(sum);
    }
    let single_thread = start.elapsed();
    println!(
        "Single thread: {:?} ({:.2} µs/iter)",
        single_thread,
        single_thread.as_micros() as f64 / n_iterations as f64
    );

    // Test 3: Barrier-synchronized approach (llama.cpp style)
    use std::sync::Arc;
    let barrier_n = Arc::new(AtomicUsize::new(0));
    let barrier_passed = Arc::new(AtomicUsize::new(0));
    let current_op = Arc::new(AtomicUsize::new(0));
    let completed = Arc::new(AtomicUsize::new(0));

    // Spawn persistent threads once
    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let barrier_n = barrier_n.clone();
            let barrier_passed = barrier_passed.clone();
            let current_op = current_op.clone();
            let completed = completed.clone();

            std::thread::spawn(move || {
                let mut last_op = 0usize;
                loop {
                    let op = current_op.load(Ordering::Acquire);
                    if op == usize::MAX {
                        break;
                    }
                    if op == last_op {
                        std::hint::spin_loop();
                        continue;
                    }
                    last_op = op;

                    // Do work
                    let mut sum = 0u64;
                    for i in 0..100 {
                        sum += i;
                    }
                    std::hint::black_box(sum);

                    // Barrier
                    let n_passed = barrier_passed.load(Ordering::Relaxed);
                    let n_barrier = barrier_n.fetch_add(1, Ordering::SeqCst);
                    if n_barrier == n_threads - 1 {
                        barrier_n.store(0, Ordering::Relaxed);
                        barrier_passed.fetch_add(1, Ordering::SeqCst);
                    } else {
                        while barrier_passed.load(Ordering::Relaxed) == n_passed
                        {
                            std::hint::spin_loop();
                        }
                    }

                    completed.fetch_add(1, Ordering::Release);
                }
            })
        })
        .collect();

    let start = Instant::now();
    for _ in 0..n_iterations {
        barrier_n.store(0, Ordering::Relaxed);
        barrier_passed.store(0, Ordering::Relaxed);
        completed.store(0, Ordering::Relaxed);

        // Signal workers
        current_op.fetch_add(1, Ordering::Release);

        // Wait for completion
        while completed.load(Ordering::Acquire) < n_threads {
            std::hint::spin_loop();
        }
    }
    let persistent = start.elapsed();

    // Shutdown
    current_op.store(usize::MAX, Ordering::Release);
    for h in handles {
        let _ = h.join();
    }

    println!(
        "Persistent threads + barrier: {:?} ({:.2} µs/iter)",
        persistent,
        persistent.as_micros() as f64 / n_iterations as f64
    );

    // Test 4: Work stealing (single atomic counter)
    let start = Instant::now();
    for _ in 0..n_iterations {
        let counter = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    loop {
                        let i = counter.fetch_add(1, Ordering::Relaxed);
                        if i >= 100 {
                            break;
                        }
                        let mut sum = 0usize;
                        sum += i;
                        std::hint::black_box(sum);
                    }
                });
            }
        });
    }
    let work_stealing = start.elapsed();
    println!(
        "Work stealing: {:?} ({:.2} µs/iter)",
        work_stealing,
        work_stealing.as_micros() as f64 / n_iterations as f64
    );

    println!("\nOverhead comparison:");
    println!(
        "Thread creation overhead: {:.2} µs per dispatch",
        create_each_time.as_micros() as f64 / n_iterations as f64
            - single_thread.as_micros() as f64 / n_iterations as f64
    );
    println!(
        "Persistent threads speedup: {:.2}x",
        create_each_time.as_secs_f64() / persistent.as_secs_f64()
    );

    // For a forward pass with 196 GEMV operations:
    let n_gemv = 196;
    println!("\nEstimated overhead per forward pass ({} GEMVs):", n_gemv);
    println!(
        "  Create each time: {:.2} ms",
        (create_each_time.as_micros() as f64 / n_iterations as f64)
            * n_gemv as f64
            / 1000.0
    );
    println!(
        "  Persistent: {:.2} ms",
        (persistent.as_micros() as f64 / n_iterations as f64) * n_gemv as f64
            / 1000.0
    );
}
