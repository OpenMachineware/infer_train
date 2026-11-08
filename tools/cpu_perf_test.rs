// Quick CPU performance test
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let mut ctx = InferenceContext::from_gguf(&args[1], Device::Cpu)
        .expect("Failed to load model");

    println!(
        "Model: {} layers, {} heads, head_dim {}",
        ctx.model.layers.len(),
        ctx.config.n_heads,
        ctx.config.head_dim
    );

    // Test decode performance
    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Prefill with 64 tokens
    let prefill_tokens: Vec<u32> = vec![1; 64];
    let start = Instant::now();
    ctx.forward(&prefill_tokens, &mut kv).expect("Prefill failed");
    let prefill_time = start.elapsed();
    let prefill_tps = 64.0 / prefill_time.as_secs_f64();

    println!("\n=== Prefill (64 tokens) ===");
    println!("Time: {:?}", prefill_time);
    println!("Speed: {:.2} t/s", prefill_tps);

    // Test decode performance
    let n_decode = 20;
    let mut times = Vec::new();

    // Warmup
    for _ in 0..3 {
        let _ = ctx.forward(&[100], &mut kv);
    }

    // Measure
    for _ in 0..n_decode {
        let start = Instant::now();
        let _ = ctx.forward(&[100], &mut kv);
        times.push(start.elapsed());
    }

    let avg_decode: std::time::Duration =
        times.iter().sum::<std::time::Duration>() / times.len() as u32;
    let decode_tps = 1.0 / avg_decode.as_secs_f64();

    println!("\n=== Decode (avg of {} runs) ===", n_decode);
    println!("Time: {:?}", avg_decode);
    println!("Speed: {:.2} t/s", decode_tps);

    println!("\n=== Comparison with llama.cpp (1.5B Q5_K_M) ===");
    println!("llama.cpp prefill: 200.95 t/s");
    println!("llama.cpp decode: 55.37 t/s");
    println!(
        "Our prefill: {:.2} t/s ({:.1}%)",
        prefill_tps,
        prefill_tps / 200.95 * 100.0
    );
    println!(
        "Our decode: {:.2} t/s ({:.1}%)",
        decode_tps,
        decode_tps / 55.37 * 100.0
    );
    println!("\nPrefill gap: {:.2}x", 200.95 / prefill_tps);
    println!("Decode gap: {:.2}x", 55.37 / decode_tps);
}
