// 插桩分析 IQ4_NL vs llama.cpp
// 分5步测量每步的绝对时间

use std::arch::aarch64::*;
use std::time::Instant;

fn main() {
    let num_heads = 32;
    let head_dim = 128;
    let seq_len = 512;

    // 初始化数据
    let q: Vec<f32> =
        (0..num_heads * head_dim).map(|i| (i % 100) as f32 / 100.0).collect();
    let mut output = vec![0.0f32; num_heads * head_dim];

    // 创建K/V缓存
    let blocks_per_row = head_dim / 32;
    let total_blocks = seq_len * num_heads * blocks_per_row;

    let mut k_cache: Vec<infer_train::quant::kv_cache::BlockIQ4NL> = vec![];
    let mut v_cache: Vec<infer_train::quant::kv_cache::BlockIQ4NL> = vec![];

    for _ in 0..total_blocks {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((j * 17) % 256) as u8;
        }
        k_cache
            .push(infer_train::quant::kv_cache::BlockIQ4NL { d: 0x3C00, qs });
        v_cache
            .push(infer_train::quant::kv_cache::BlockIQ4NL { d: 0x3C00, qs });
    }

    // 预热
    for _ in 0..3 {
        unsafe {
            infer_train::ops::attention::decode_attention_iq4_nl_neon(
                &q,
                &k_cache,
                &v_cache,
                &mut output,
                num_heads,
                head_dim,
                seq_len,
                0.125,
            );
        }
    }

    // 插桩测量
    let iterations = 20;
    let mut time_quantize = 0u64;
    let mut time_vec_dot = 0u64;
    let mut time_softmax = 0u64;
    let mut time_vec_mad = 0u64;
    let mut time_normalize = 0u64;

    for _ in 0..iterations {
        let start_total = Instant::now();

        unsafe {
            profile_iq4_nl_attention(
                &q,
                &k_cache,
                &v_cache,
                &mut output,
                num_heads,
                head_dim,
                seq_len,
                0.125,
                &mut time_quantize,
                &mut time_vec_dot,
                &mut time_softmax,
                &mut time_vec_mad,
                &mut time_normalize,
            );
        }
    }

    println!("=== IQ4_NL Attention Profiling (seq_len={}) ===", seq_len);
    println!(
        "A. Quantize Q to Q8_0:   {:8} ns ({}%)",
        time_quantize / iterations,
        time_quantize * 100
            / (time_quantize
                + time_vec_dot
                + time_softmax
                + time_vec_mad
                + time_normalize)
    );
    println!(
        "B. Q × K vec_dot:        {:8} ns ({}%)",
        time_vec_dot / iterations,
        time_vec_dot * 100
            / (time_quantize
                + time_vec_dot
                + time_softmax
                + time_vec_mad
                + time_normalize)
    );
    println!(
        "C. Softmax + scaling:    {:8} ns ({}%)",
        time_softmax / iterations,
        time_softmax * 100
            / (time_quantize
                + time_vec_dot
                + time_softmax
                + time_vec_mad
                + time_normalize)
    );
    println!(
        "D. weight × V vec_mad:   {:8} ns ({}%)",
        time_vec_mad / iterations,
        time_vec_mad * 100
            / (time_quantize
                + time_vec_dot
                + time_softmax
                + time_vec_mad
                + time_normalize)
    );
    println!(
        "E. Normalize output:     {:8} ns ({}%)",
        time_normalize / iterations,
        time_normalize * 100
            / (time_quantize
                + time_vec_dot
                + time_softmax
                + time_vec_mad
                + time_normalize)
    );
    println!(
        "Total:                   {:8} ns",
        (time_quantize
            + time_vec_dot
            + time_softmax
            + time_vec_mad
            + time_normalize)
            / iterations
    );

    // 计算GFLOPS
    let total_time_ns = (time_quantize
        + time_vec_dot
        + time_softmax
        + time_vec_mad
        + time_normalize)
        / iterations;
    let total_flops = 2 * seq_len * head_dim * num_heads; // Q×K + V×output
    let gflops = total_flops as f64 / (total_time_ns as f64 / 1e9) / 1e9;
    println!("\nGFLOPS: {:.2}", gflops);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn profile_iq4_nl_attention(
    q: &[f32],
    k: &[infer_train::quant::kv_cache::BlockIQ4NL],
    v: &[infer_train::quant::kv_cache::BlockIQ4NL],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    time_quantize: &mut u64,
    time_vec_dot: &mut u64,
    time_softmax: &mut u64,
    time_vec_mad: &mut u64,
    time_normalize: &mut u64,
) {
    use infer_train::ops::attention_fused::{
        quantize_query_to_q8_0, vec_mad_f32_iq4_nl_neon,
    };
    use infer_train::quant::kv_cache::{BlockQ8_0, QK};

    let blocks_per_row = head_dim / QK;

    let mut q_q8 = vec![BlockQ8_0 { d: 0, qs: [0i8; QK] }; blocks_per_row];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        // A. 量化Q
        let t0 = std::time::Instant::now();
        quantize_query_to_q8_0(q_head, &mut q_q8);
        *time_quantize += t0.elapsed().as_nanos() as u64;

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for d in 0..head_dim {
            output_head[d] = 0.0;
        }

        for s in 0..seq_len {
            let k_offset = (s * num_heads + h) * blocks_per_row;

            // B. Q × K vec_dot (直接调用attention.rs中的函数)
            let t1 = std::time::Instant::now();
            let q_types = std::slice::from_raw_parts(
                q_q8.as_ptr() as *const infer_train::quant::types::BlockQ8_0,
                q_q8.len(),
            );
            let k_types = std::slice::from_raw_parts(
                k.as_ptr().add(k_offset)
                    as *const infer_train::quant::types::BlockIQ4NL,
                blocks_per_row,
            );
            let score =
                infer_train::quant::vec_dot::arm::iq::vec_dot_iq4_nl_q8_0_neon(
                    head_dim, k_types, q_types,
                ) * scale;
            *time_vec_dot += t1.elapsed().as_nanos() as u64;

            // C. Softmax + scaling
            let t2 = std::time::Instant::now();
            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                let corr_vec = vdupq_n_f32(correction);
                for d in (0..head_dim).step_by(4) {
                    if d + 4 <= head_dim {
                        let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                        let scaled = vmulq_f32(out_vec, corr_vec);
                        vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
                    } else {
                        for dd in d..head_dim {
                            output_head[dd] *= correction;
                        }
                    }
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }
            sum_exp += weight;
            *time_softmax += t2.elapsed().as_nanos() as u64;

            // D. weight × V vec_mad
            let t3 = std::time::Instant::now();
            let v_offset = (s * num_heads + h) * blocks_per_row;
            vec_mad_f32_iq4_nl_neon(
                output_head,
                &v[v_offset..v_offset + blocks_per_row],
                weight,
                head_dim,
            );
            *time_vec_mad += t3.elapsed().as_nanos() as u64;
        }

        // E. 归一化
        let t4 = std::time::Instant::now();
        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            if d + 4 <= head_dim {
                let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                let scaled = vmulq_f32(out_vec, vdupq_n_f32(inv_sum));
                vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
            } else {
                for dd in d..head_dim {
                    output_head[dd] *= inv_sum;
                }
            }
        }
        *time_normalize += t4.elapsed().as_nanos() as u64;
    }
}
