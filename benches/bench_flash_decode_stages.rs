// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Stage-by-stage profiling of flash decode

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 500;
const SEQ_K: usize = 1024;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../src/ops/metal/flash_decode_profiling.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Flash Decode Stage Profiling ===");
    println!(
        "Config: {} heads, {} head_dim, seq_k={}, {} rounds\n",
        NUM_HEADS, HEAD_DIM, SEQ_K, NUM_ROUNDS
    );

    // Create data
    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; SEQ_K * NUM_HEADS * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3400; SEQ_K * NUM_HEADS * HEAD_DIM];

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
    let scores_buf = device.new_buffer(
        (SEQ_K * NUM_HEADS * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Get pipelines
    let stage1 = library.get_function("flash_decode_stage1_qk", None).unwrap();
    let stage2 =
        library.get_function("flash_decode_stage2_softmax", None).unwrap();
    let stage3 =
        library.get_function("flash_decode_stage3_vacc", None).unwrap();

    let pipeline1 =
        device.new_compute_pipeline_state_with_function(&stage1).unwrap();
    let pipeline2 =
        device.new_compute_pipeline_state_with_function(&stage2).unwrap();
    let pipeline3 =
        device.new_compute_pipeline_state_with_function(&stage3).unwrap();

    // Profile each stage
    let t1 = profile_stage(&device, &queue, &pipeline1, NUM_ROUNDS, || {
        run_stage1(&queue, &pipeline1, &q_buf, &k_buf, &scores_buf);
    });

    let t2 = profile_stage(&device, &queue, &pipeline2, NUM_ROUNDS, || {
        run_stage2(&queue, &pipeline2, &scores_buf);
    });

    let t3 = profile_stage(&device, &queue, &pipeline3, NUM_ROUNDS, || {
        run_stage3(&queue, &pipeline3, &scores_buf, &v_buf, &out_buf);
    });

    println!("Stage-by-stage timing (median, {} rounds):", NUM_ROUNDS);
    println!("  Stage 1 (Q*K^T):    {:>8.2} µs", t1);
    println!("  Stage 2 (Softmax):  {:>8.2} µs", t2);
    println!("  Stage 3 (V accum):  {:>8.2} µs", t3);
    println!("  Total:              {:>8.2} µs", t1 + t2 + t3);
    println!();

    // Calculate GFLOPS per stage
    let total_flops = 4.0 * SEQ_K as f64 * HEAD_DIM as f64 * NUM_HEADS as f64;
    let qk_flops = 2.0 * SEQ_K as f64 * HEAD_DIM as f64 * NUM_HEADS as f64; // Q*K^T
    let softmax_flops = SEQ_K as f64 * NUM_HEADS as f64 * 10.0; // exp, max, sum
    let vacc_flops = 2.0 * SEQ_K as f64 * HEAD_DIM as f64 * NUM_HEADS as f64; // V*output

    println!("GFLOPS per stage:");
    println!(
        "  Stage 1: {:>8.2} G (expected: {:.2} G)",
        qk_flops / t1 / 1000.0,
        qk_flops / total_flops * 48.59
    );
    println!("  Stage 2: {:>8.2} G", softmax_flops / t2 / 1000.0);
    println!(
        "  Stage 3: {:>8.2} G (expected: {:.2} G)",
        vacc_flops / t3 / 1000.0,
        vacc_flops / total_flops * 48.59
    );
    println!();

    // Compare with combined kernel
    let combined_source =
        include_str!("../src/ops/metal/flash_decode_vec.metal");
    let combined_lib =
        device.new_library_with_source(combined_source, &options).unwrap();
    let combined_fn =
        combined_lib.get_function("flash_decode_vec_f16", None).unwrap();
    let combined_pipeline =
        device.new_compute_pipeline_state_with_function(&combined_fn).unwrap();

    let t_combined =
        profile_stage(&device, &queue, &combined_pipeline, NUM_ROUNDS, || {
            run_combined(
                &queue,
                &combined_pipeline,
                &q_buf,
                &k_buf,
                &v_buf,
                &out_buf,
            );
        });

    println!("Comparison:");
    println!("  Staged total:   {:>8.2} µs", t1 + t2 + t3);
    println!("  Combined:       {:>8.2} µs", t_combined);
    println!(
        "  Overhead:       {:>8.2} µs ({:.1}%)",
        (t1 + t2 + t3) - t_combined,
        ((t1 + t2 + t3) / t_combined - 1.0) * 100.0
    );
}

fn profile_stage<F>(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    rounds: usize,
    mut run: F,
) -> f64
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10 {
        run();
    }

    // Measure
    let mut times = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t0 = Instant::now();
        run();
        times.push(t0.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[rounds / 2] / 1000.0 // µs
}

fn run_stage1(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    q: &Buffer,
    k: &Buffer,
    scores: &Buffer,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);

    let args = [SEQ_K as u32, (1.0f32 / (128.0f32).sqrt()).to_bits()];
    enc.set_bytes(0, 8, args.as_ptr() as *const _);
    enc.set_buffer(2, Some(q), 0);
    enc.set_buffer(3, Some(k), 0);
    enc.set_buffer(4, Some(scores), 0);

    enc.dispatch_thread_groups(
        MTLSize::new(1, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn run_stage2(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    scores: &Buffer,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);

    let args = [SEQ_K as u32];
    enc.set_bytes(0, 4, args.as_ptr() as *const _);
    enc.set_buffer(1, Some(scores), 0);

    enc.dispatch_thread_groups(
        MTLSize::new(1, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn run_stage3(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    scores: &Buffer,
    v: &Buffer,
    out: &Buffer,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);

    let args = [SEQ_K as u32];
    enc.set_bytes(0, 4, args.as_ptr() as *const _);
    enc.set_buffer(1, Some(scores), 0);
    enc.set_buffer(2, Some(v), 0);
    enc.set_buffer(3, Some(out), 0);

    enc.dispatch_thread_groups(
        MTLSize::new(1, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn run_combined(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    q: &Buffer,
    k: &Buffer,
    v: &Buffer,
    out: &Buffer,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);

    let args = [SEQ_K as u32, (1.0f32 / (128.0f32).sqrt()).to_bits()];
    enc.set_bytes(0, 8, args.as_ptr() as *const _);
    enc.set_buffer(2, Some(q), 0);
    enc.set_buffer(3, Some(k), 0);
    enc.set_buffer(4, Some(v), 0);
    enc.set_buffer(5, Some(out), 0);
    enc.set_threadgroup_memory_length(0, (128 + 32 + 128) * 4);

    enc.dispatch_thread_groups(
        MTLSize::new(1, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}
