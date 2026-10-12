// Correctness test for self-written flash attention kernels
// Compares GPU output against CPU reference implementation

use metal::*;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Test configurations: (name, batch, heads, seq_q, seq_k, head_dim)
    let configs = [
        ("Decode 32", 1, 1, 1, 32, 64),
        ("Decode 64", 1, 1, 1, 64, 64),
        ("Decode 128", 1, 1, 1, 128, 64),
        ("Decode 256", 1, 1, 1, 256, 64),
        ("Decode 512", 1, 1, 1, 512, 64),
        ("Prefill 4", 1, 1, 4, 32, 64),
        ("Prefill 16", 1, 1, 16, 64, 64),
        ("Prefill 64", 1, 1, 64, 128, 64),
    ];

    println!("=== Flash Attention Correctness Test ===\n");

    let mut seed: u64 = 12345;

    for (name, batch, heads, seq_q, seq_k, head_dim) in configs {
        println!("Testing: {}", name);
        println!(
            "  batch={}, heads={}, seq_q={}, seq_k={}, head_dim={}",
            batch, heads, seq_q, seq_k, head_dim
        );

        // Create pseudo-random data using XOR-shift
        let q_size = batch * heads * seq_q * head_dim;
        let k_size = batch * heads * seq_k * head_dim;
        let v_size = batch * heads * seq_k * head_dim;

        let q_data: Vec<f32> = (0..q_size).map(|_| rng(&mut seed)).collect();
        let k_data: Vec<f32> = (0..k_size).map(|_| rng(&mut seed)).collect();
        let v_data: Vec<f32> = (0..v_size).map(|_| rng(&mut seed)).collect();

        // CPU reference implementation
        let cpu_output = cpu_attention(
            &q_data, &k_data, &v_data, batch, heads, seq_q, seq_k, head_dim,
        );

        // GPU implementation
        let gpu_output = run_gpu_flash_attention(
            &device, &queue, &q_data, &k_data, &v_data, batch, heads, seq_q,
            seq_k, head_dim,
        );

        // Compare
        let max_diff = compare_outputs(&cpu_output, &gpu_output);
        let avg_diff: f64 = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / cpu_output.len() as f64;

        println!("  Max diff: {:.6e}", max_diff);
        println!("  Avg diff: {:.6e}", avg_diff);

        if max_diff > 1e-4 {
            println!("  First 10 values:");
            for i in 0..10.min(cpu_output.len()) {
                println!(
                    "    [{}] CPU: {:.6}, GPU: {:.6}, diff: {:.6}",
                    i,
                    cpu_output[i],
                    gpu_output[i],
                    (cpu_output[i] - gpu_output[i]).abs()
                );
            }
        }

        println!("  Status: {}", if max_diff < 1e-4 { "PASS" } else { "FAIL" });
        println!();
    }
}

fn rng(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * heads * seq_q * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for b in 0..batch {
        for h in 0..heads {
            for sq in 0..seq_q {
                // Get query vector
                let q_offset = b * heads * seq_q * head_dim
                    + h * seq_q * head_dim
                    + sq * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                // Compute attention scores
                let mut scores = vec![0.0f32; seq_k];
                for sk in 0..seq_k {
                    let k_offset = b * heads * seq_k * head_dim
                        + h * seq_k * head_dim
                        + sk * head_dim;
                    let k_vec = &k[k_offset..k_offset + head_dim];

                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q_vec[d] * k_vec[d];
                    }
                    scores[sk] = score * scale;
                }

                // Softmax
                let max_score =
                    scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 =
                    scores.iter().map(|s| (s - max_score).exp()).sum();

                // Compute weighted output
                let o_offset = b * heads * seq_q * head_dim
                    + h * seq_q * head_dim
                    + sq * head_dim;
                for sk in 0..seq_k {
                    let weight = (scores[sk] - max_score).exp() / exp_sum;
                    let v_offset = b * heads * seq_k * head_dim
                        + h * seq_k * head_dim
                        + sk * head_dim;
                    let v_vec = &v[v_offset..v_offset + head_dim];

                    for d in 0..head_dim {
                        output[o_offset + d] += weight * v_vec[d];
                    }
                }
            }
        }
    }

    output
}

fn run_gpu_flash_attention(
    device: &Device,
    queue: &CommandQueue,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Compile kernel
    let source = include_str!("../src/ops/metal/flash_attn.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");

    // Choose kernel based on seq_q
    let kernel_name = if seq_q == 1 {
        "flash_decode_attention_f32"
    } else {
        "flash_prefill_attention_f32"
    };
    let kernel =
        library.get_function(kernel_name, None).expect("Kernel not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Pipeline failed");

    // Create buffers
    let q_buf = device.new_buffer(
        (q.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer(
        (k.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer(
        (v.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (q.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            q.as_ptr(),
            q_buf.contents() as *mut f32,
            q.len(),
        );
        std::ptr::copy_nonoverlapping(
            k.as_ptr(),
            k_buf.contents() as *mut f32,
            k.len(),
        );
        std::ptr::copy_nonoverlapping(
            v.as_ptr(),
            v_buf.contents() as *mut f32,
            v.len(),
        );
    }

    // Arguments
    #[repr(C)]
    struct Args {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        scale: f32,
        q_stride_b: u32,
        q_stride_h: u32,
        q_stride_s: u32,
        k_stride_b: u32,
        k_stride_h: u32,
        k_stride_s: u32,
        v_stride_b: u32,
        v_stride_h: u32,
        v_stride_s: u32,
        o_stride_b: u32,
        o_stride_h: u32,
        o_stride_s: u32,
    }

    let args = Args {
        batch: batch as u32,
        heads: heads as u32,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        q_stride_b: (heads * seq_q * head_dim) as u32,
        q_stride_h: (seq_q * head_dim) as u32,
        q_stride_s: head_dim as u32,
        k_stride_b: (heads * seq_k * head_dim) as u32,
        k_stride_h: (seq_k * head_dim) as u32,
        k_stride_s: head_dim as u32,
        v_stride_b: (heads * seq_k * head_dim) as u32,
        v_stride_h: (seq_k * head_dim) as u32,
        v_stride_s: head_dim as u32,
        o_stride_b: (heads * seq_q * head_dim) as u32,
        o_stride_h: (seq_q * head_dim) as u32,
        o_stride_s: head_dim as u32,
    };

    // Dispatch
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<Args>() as u64,
        &args as *const _ as *const _,
    );
    enc.set_buffer(1, Some(&q_buf), 0);
    enc.set_buffer(2, Some(&k_buf), 0);
    enc.set_buffer(3, Some(&v_buf), 0);
    enc.set_buffer(4, Some(&out_buf), 0);

    if seq_q == 1 {
        // Decode: one threadgroup per (batch, head)
        // Fixed threadgroup size of 128 threads, each processes stride of KV
        let grid_size =
            MTLSize { width: 1, height: heads as u64, depth: batch as u64 };
        let threadgroup_size = MTLSize { width: 128, height: 1, depth: 1 };
        enc.dispatch_thread_groups(grid_size, threadgroup_size);
    } else {
        // Prefill: one threadgroup per (query, head, batch)
        let grid_size = MTLSize {
            width: seq_q as u64,
            height: heads as u64,
            depth: batch as u64,
        };
        let threadgroup_size = MTLSize { width: 128, height: 1, depth: 1 };
        enc.dispatch_thread_groups(grid_size, threadgroup_size);
    }

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Read output
    let mut output = vec![0.0f32; q.len()];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents() as *const f32,
            output.as_mut_ptr(),
            output.len(),
        );
    }

    output
}

fn compare_outputs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}
