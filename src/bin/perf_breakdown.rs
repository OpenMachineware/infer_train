use infer_train::model::GgufFile;

fn main() {
    let path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(path).unwrap();

    println!("=== Performance Breakdown ===\n");

    let layer_weights = [
        ("blk.0.attn_q.weight", "Q projection"),
        ("blk.0.attn_k.weight", "K projection"),
        ("blk.0.attn_v.weight", "V projection"),
        ("blk.0.attn_output.weight", "Output projection"),
        ("blk.0.ffn_gate.weight", "Gate projection"),
        ("blk.0.ffn_up.weight", "Up projection"),
        ("blk.0.ffn_down.weight", "Down projection"),
    ];

    let mut total_layer_bytes = 0usize;

    for (name, desc) in &layer_weights {
        if let Some(tensor) = gguf.tensors.iter().find(|t| t.name == *name) {
            let bytes = tensor.strides[1] as usize * tensor.dims[1] as usize;
            total_layer_bytes += bytes;
            let in_f = tensor.dims[0] as usize;
            let out_f = tensor.dims[1] as usize;
            println!(
                "{}: {}x{} ({:?}), {} KB",
                desc,
                in_f,
                out_f,
                tensor.quant_type,
                bytes / 1024
            );
        }
    }

    println!("\nTotal per layer: {} KB", total_layer_bytes / 1024);

    // Output projection
    if let Some(tensor) =
        gguf.tensors.iter().find(|t| t.name == "token_embd.weight")
    {
        let bytes = tensor.strides[1] as usize * tensor.dims[1] as usize;
        println!("\nOutput projection (token_embd): {} KB", bytes / 1024);
    }

    println!("\n=== Time Estimation ===");
    println!("Per layer GEMV time (at 17 GB/s):");
    let layer_time_us = (total_layer_bytes as f64 / 17e9 * 1e6) as f64;
    println!("  ~{:.2} μs", layer_time_us);

    println!("\n28 layers: {:.2} μs", layer_time_us * 28.0);

    println!(
        "\nWith output projection: + {:.2} μs",
        (151936 * 840) as f64 / 17e9 * 1e6
    );

    // Total estimated time
    let total_bytes = total_layer_bytes * 28 + 151936 * 840;
    let total_time_us = total_bytes as f64 / 17e9 * 1e6;
    println!("\nTotal weight bytes: {} MB", total_bytes / 1024 / 1024);
    println!("Estimated time: {:.2} μs", total_time_us);
    println!("Estimated throughput: {:.1} t/s", 1e6 / total_time_us);

    println!("\n=== Actual Measurement ===");
    println!("Measured: ~60 ms/token, 16 t/s");
    println!("Gap: {:.1}x", 60000.0 / total_time_us);
}
