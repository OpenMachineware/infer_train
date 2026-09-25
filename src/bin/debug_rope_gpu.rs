// Debug GPU RoPE correctness
use infer_train::ops::gpu::metal::RopeMetalContext;
use infer_train::ops::rope::rope_neox_simple;

fn main() {
    println!("=== GPU RoPE Correctness Debug ===\n");

    let gpu_ctx = RopeMetalContext::new().unwrap();

    let n_heads = 1;
    let hidden_dim = 4;
    let n_dims = 4;
    let n_seqs = 1;

    // Simple test data
    let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let positions: Vec<i32> = vec![1];

    println!("Input: {:?}", src);

    let dst_gpu = gpu_ctx.rope_neox_f32(&src, &positions, n_heads, hidden_dim, n_dims, 10000.0).unwrap();
    println!("GPU output: {:?}", dst_gpu);

    let mut dst_cpu = vec![0.0f32; hidden_dim];
    rope_neox_simple(&src, &mut dst_cpu, positions[0] as usize, n_dims, 10000.0);
    println!("CPU output: {:?}", dst_cpu);

    for i in 0..n_dims {
        let diff = (dst_gpu[i] - dst_cpu[i]).abs();
        println!("  [{}] GPU={:.6} CPU={:.6} diff={:.2e}", i, dst_gpu[i], dst_cpu[i], diff);
    }
}
