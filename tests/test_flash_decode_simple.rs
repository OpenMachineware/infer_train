use metal::*;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../src/ops/metal/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    let kernel =
        library.get_function("flash_decode_attention_f16", None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    let num_heads = 32;
    let head_dim = 128;
    let seq_k = 512;

    // F16 data
    let q_data: Vec<u16> = vec![0x3C00; num_heads * head_dim]; // 1.0
    let k_data: Vec<u16> = vec![0x3800; seq_k * num_heads * head_dim]; // 0.5
    let v_data: Vec<u16> = vec![0x3400; seq_k * num_heads * head_dim]; // 0.3

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_data.as_ptr() as *const _,
        (k_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_data.as_ptr() as *const _,
        (v_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (num_heads * head_dim * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

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
        batch: 1,
        heads: num_heads as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        q_stride_b: (num_heads * head_dim) as u32,
        q_stride_h: head_dim as u32,
        q_stride_s: head_dim as u32,
        k_stride_b: (num_heads * seq_k * head_dim * 2) as u32,
        k_stride_h: (seq_k * head_dim * 2) as u32,
        k_stride_s: (head_dim * 2) as u32,
        v_stride_b: (num_heads * seq_k * head_dim * 2) as u32,
        v_stride_h: (seq_k * head_dim * 2) as u32,
        v_stride_s: (head_dim * 2) as u32,
        o_stride_b: (num_heads * head_dim) as u32,
        o_stride_h: head_dim as u32,
        o_stride_s: head_dim as u32,
    };

    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<Args>(),
        )
    };

    // Warmup
    for _ in 0..10 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_bytes(
            0,
            args_bytes.len() as u64,
            args_bytes.as_ptr() as *const _,
        );
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&k_buf), 0);
        enc.set_buffer(3, Some(&v_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(1, num_heads as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark
    use std::time::Instant;
    let start = Instant::now();
    for _ in 0..100 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_bytes(
            0,
            args_bytes.len() as u64,
            args_bytes.as_ptr() as *const _,
        );
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&k_buf), 0);
        enc.set_buffer(3, Some(&v_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(1, num_heads as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let flops = 2.0 * num_heads as f64 * seq_k as f64 * head_dim as f64 * 100.0;
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    println!("Total time: {:?}", elapsed);
    println!("Per iteration: {:?}", elapsed / 100);
    println!("GFLOPS: {:.2}", gflops);

    // Read output to verify
    let output: &[f32] = unsafe {
        std::slice::from_raw_parts(
            out_buf.contents() as *const f32,
            num_heads * head_dim,
        )
    };
    println!("First 4 output values: {:?}", &output[0..4]);
    println!("Expected (all should be similar): ~0.17");
}
