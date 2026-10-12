// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Test correctness: Adaptive vs Multi-WG must produce same results

use metal::*;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NWG: usize = 32;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../../shaders/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Flash Decode Correctness Test ===\n");

    // Test different sequence lengths
    for seq_k in [64, 128, 256, 512, 1024, 2048] {
        println!("Testing seq_k = {}", seq_k);

        let adapt_result = run_adaptive(&device, &queue, &library, seq_k);
        let multi_wg_result = run_multi_wg(&device, &queue, &library, seq_k);

        // Compare results
        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f32;
        for i in 0..adapt_result.len() {
            let diff = (adapt_result[i] - multi_wg_result[i]).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff;
        }
        let avg_diff = total_diff / adapt_result.len() as f32;

        println!("  Max diff: {:.6e}, Avg diff: {:.6e}", max_diff, avg_diff);

        if max_diff < 1e-4 {
            println!("  ✓ PASS\n");
        } else {
            println!("  ✗ FAIL - results differ!\n");

            // Show sample outputs
            println!("  Sample outputs (first head, first 8 dims):");
            println!("    Adaptive: {:?}", &adapt_result[..8]);
            println!("    Multi-WG: {:?}", &multi_wg_result[..8]);
            return;
        }
    }

    println!("All correctness tests passed!");
}

fn run_adaptive(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    seq_k: usize,
) -> Vec<f32> {
    // Create simple test data
    let q_data: Vec<f32> =
        (0..NUM_HEADS * HEAD_DIM).map(|i| (i % 100) as f32 / 100.0).collect();

    // F16 KV cache - all ones for simplicity
    let n_elements = seq_k * NUM_HEADS * HEAD_DIM;
    let kv_data: Vec<u16> = vec![0x3C00; n_elements]; // 1.0 in FP16

    // NSG based on seq_k
    let nsg = if seq_k <= 128 {
        1
    } else if seq_k <= 256 {
        2
    } else if seq_k <= 512 {
        4
    } else {
        8
    };

    let args = FlashAttnArgs {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        q_stride_b: (NUM_HEADS * HEAD_DIM * 4) as u32,
        q_stride_h: (HEAD_DIM * 4) as u32,
        q_stride_s: (HEAD_DIM * 4) as u32,
        k_stride_b: (NUM_HEADS * seq_k * 2) as u32,
        k_stride_h: (seq_k * 2) as u32,
        k_stride_s: 2,
        v_stride_b: (NUM_HEADS * seq_k * 2) as u32,
        v_stride_h: (seq_k * 2) as u32,
        v_stride_s: 2,
        o_stride_b: (NUM_HEADS * HEAD_DIM * 4) as u32,
        o_stride_h: (HEAD_DIM * 4) as u32,
        o_stride_s: (HEAD_DIM * 4) as u32,
    };

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        kv_data.as_ptr() as *const _,
        (kv_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        kv_data.as_ptr() as *const _,
        (kv_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel = library
        .get_function("flash_decode_attention_f16_adaptive", None)
        .unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    let shmem_size = nsg * (2 + HEAD_DIM) * 4;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<FlashAttnArgs>() as u64,
        &args as *const _ as *const _,
    );
    enc.set_buffer(1, Some(&q_buf), 0);
    enc.set_buffer(2, Some(&k_buf), 0);
    enc.set_buffer(3, Some(&v_buf), 0);
    enc.set_buffer(4, Some(&out_buf), 0);
    enc.set_threadgroup_memory_length(0, shmem_size as u64);
    enc.dispatch_thread_groups(
        MTLSize::new(1, NUM_HEADS as u64, 1),
        MTLSize::new(32, nsg as u64, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = out_buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, NUM_HEADS * HEAD_DIM).to_vec() }
}

fn run_multi_wg(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    seq_k: usize,
) -> Vec<f32> {
    // Same test data
    let q_data: Vec<f32> =
        (0..NUM_HEADS * HEAD_DIM).map(|i| (i % 100) as f32 / 100.0).collect();
    let n_elements = seq_k * NUM_HEADS * HEAD_DIM;
    let kv_data: Vec<u16> = vec![0x3C00; n_elements]; // 1.0 in FP16

    let args = FlashAttnArgs {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        q_stride_b: (NUM_HEADS * HEAD_DIM * 4) as u32,
        q_stride_h: (HEAD_DIM * 4) as u32,
        q_stride_s: (HEAD_DIM * 4) as u32,
        k_stride_b: (NUM_HEADS * seq_k * 2) as u32,
        k_stride_h: (seq_k * 2) as u32,
        k_stride_s: 2,
        v_stride_b: (NUM_HEADS * seq_k * 2) as u32,
        v_stride_h: (seq_k * 2) as u32,
        v_stride_s: 2,
        o_stride_b: (NUM_HEADS * HEAD_DIM * 4) as u32,
        o_stride_h: (HEAD_DIM * 4) as u32,
        o_stride_s: (HEAD_DIM * 4) as u32,
    };

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        kv_data.as_ptr() as *const _,
        (kv_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        kv_data.as_ptr() as *const _,
        (kv_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_out_buf = device.new_buffer(
        (NWG * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_ms_buf = device
        .new_buffer((NWG * 8) as u64, MTLResourceOptions::StorageModeShared);

    let multi_wg_kernel = library
        .get_function("flash_decode_attention_f16_multi_wg", None)
        .unwrap();
    let multi_wg_pipeline = device
        .new_compute_pipeline_state_with_function(&multi_wg_kernel)
        .unwrap();

    let reduce_kernel =
        library.get_function("flash_decode_attention_reduce", None).unwrap();
    let reduce_pipeline = device
        .new_compute_pipeline_state_with_function(&reduce_kernel)
        .unwrap();

    let cmd = queue.new_command_buffer();

    // Phase 1: Multi-workgroup attention
    let enc1 = cmd.new_compute_command_encoder();
    enc1.set_compute_pipeline_state(&multi_wg_pipeline);
    enc1.set_bytes(
        0,
        std::mem::size_of::<FlashAttnArgs>() as u64,
        &args as *const _ as *const _,
    );
    enc1.set_buffer(1, Some(&q_buf), 0);
    enc1.set_buffer(2, Some(&k_buf), 0);
    enc1.set_buffer(3, Some(&v_buf), 0);
    enc1.set_buffer(4, Some(&partial_out_buf), 0);
    enc1.set_buffer(5, Some(&partial_ms_buf), 0);
    enc1.set_threadgroup_memory_length(0, 4100);
    enc1.dispatch_thread_groups(
        MTLSize::new(NWG as u64, NUM_HEADS as u64, 1),
        MTLSize::new(32, 2, 1),
    );
    enc1.end_encoding();

    // Phase 2: Reduction
    let enc2 = cmd.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&reduce_pipeline);
    enc2.set_bytes(
        0,
        std::mem::size_of::<FlashAttnArgs>() as u64,
        &args as *const _ as *const _,
    );
    enc2.set_buffer(1, Some(&partial_out_buf), 0);
    enc2.set_buffer(2, Some(&partial_ms_buf), 0);
    enc2.set_buffer(3, Some(&out_buf), 0);
    let nwg_u32 = NWG as u32;
    enc2.set_bytes(4, 4, &nwg_u32 as *const _ as *const _);
    enc2.dispatch_thread_groups(
        MTLSize::new(1, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc2.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    let ptr = out_buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, NUM_HEADS * HEAD_DIM).to_vec() }
}

#[repr(C)]
struct FlashAttnArgs {
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
