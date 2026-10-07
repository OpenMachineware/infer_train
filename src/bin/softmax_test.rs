// Test attention kernel with multiple KV positions
use infer_train::quant::kv_cache::{quantize_row_q4_0, BlockQ4_0};
use metal::*;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("=== Attention Kernel Debug ===\n");

    let head_dim = 128;
    let seq_k = 4; // Small number for debugging
    let mut seed: u64 = 42;

    // Create Q (one row)
    let q: Vec<f32> = (0..head_dim).map(|_| rng(&mut seed)).collect();

    // Create K and V
    let k: Vec<f32> = (0..seq_k * head_dim).map(|_| rng(&mut seed)).collect();
    let v: Vec<f32> = (0..seq_k * head_dim).map(|_| rng(&mut seed)).collect();

    // Quantize K and V
    let blocks_per_row = head_dim / 32;
    let mut k_quant = Vec::new();
    let mut v_quant = Vec::new();
    for sk in 0..seq_k {
        let mut k_blocks: Vec<BlockQ4_0> = Vec::with_capacity(blocks_per_row);
        unsafe {
            k_blocks.set_len(blocks_per_row);
        }
        quantize_row_q4_0(
            &k[sk * head_dim..(sk + 1) * head_dim],
            &mut k_blocks,
        );
        let bytes = unsafe {
            std::slice::from_raw_parts(
                k_blocks.as_ptr() as *const u8,
                blocks_per_row * std::mem::size_of::<BlockQ4_0>(),
            )
        };
        k_quant.extend_from_slice(bytes);

        let mut v_blocks: Vec<BlockQ4_0> = Vec::with_capacity(blocks_per_row);
        unsafe {
            v_blocks.set_len(blocks_per_row);
        }
        quantize_row_q4_0(
            &v[sk * head_dim..(sk + 1) * head_dim],
            &mut v_blocks,
        );
        let bytes = unsafe {
            std::slice::from_raw_parts(
                v_blocks.as_ptr() as *const u8,
                blocks_per_row * std::mem::size_of::<BlockQ4_0>(),
            )
        };
        v_quant.extend_from_slice(bytes);
    }

    // CPU reference
    let scale = 1.0 / (head_dim as f32).sqrt();
    let cpu_output = cpu_attention(
        &q,
        &k_quant,
        &v_quant,
        seq_k,
        head_dim,
        blocks_per_row,
        scale,
    );

    // GPU kernel
    let source = r#"
#include <metal_stdlib>
using namespace metal;

#define QK 32

struct block_q4_0 {
    half d;
    uint8_t qs[QK / 2];
};

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.0h) : xb->d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

struct Args {
    uint32_t seq_k;
    uint32_t head_dim;
    float scale;
    uint32_t k_stride_s;
    uint32_t v_stride_s;
};

kernel void debug_attention(
    constant Args & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device const char * v [[buffer(3)]],
    device float * output [[buffer(4)]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr ushort BLK_SIZE = 32;
    constexpr ushort BLK4 = BLK_SIZE / 4;

    const ushort tid = sgitg * 32 + tiisg;
    const ushort num_threads = 32; // Single SIMD group

    // Online softmax state
    float local_max = -FLT_MAX/2;
    float local_sum = 0.0f;
    float local_out[128];
    for (ushort d = 0; d < args.head_dim; d++) {
        local_out[d] = 0.0f;
    }

    // Process KV positions
    for (uint kv_idx = tid; kv_idx < args.seq_k; kv_idx += num_threads) {
        device const char * k_row = k + kv_idx * args.k_stride_s;
        device const char * v_row = v + kv_idx * args.v_stride_s;

        // Dequantize K
        float k_deq[128];
        for (ushort b = 0; b < args.head_dim / BLK_SIZE; b++) {
            device const block_q4_0 * kb = (device const block_q4_0 *)(k_row + b * sizeof(block_q4_0));
            for (ushort il = 0; il < BLK4; il++) {
                float4 reg;
                dequantize_q4_0_t4(kb, il, reg);
                ushort d4 = b * BLK4 + il;
                k_deq[d4 * 4 + 0] = reg[0];
                k_deq[d4 * 4 + 1] = reg[1];
                k_deq[d4 * 4 + 2] = reg[2];
                k_deq[d4 * 4 + 3] = reg[3];
            }
        }

        // Q*K
        float score = 0.0f;
        for (ushort d = 0; d < args.head_dim; d++) {
            score += q[d] * k_deq[d];
        }
        score *= args.scale;

        // Online softmax
        float old_max = local_max;
        local_max = max(local_max, score);

        if (old_max > -FLT_MAX/4) {
            float scale_old = exp(old_max - local_max);
            local_sum *= scale_old;
            for (ushort d = 0; d < args.head_dim; d++) {
                local_out[d] *= scale_old;
            }
        }

        float weight = exp(score - local_max);
        local_sum += weight;

        // Dequantize V and accumulate
        for (ushort b = 0; b < args.head_dim / BLK_SIZE; b++) {
            device const block_q4_0 * vb = (device const block_q4_0 *)(v_row + b * sizeof(block_q4_0));
            for (ushort il = 0; il < BLK4; il++) {
                float4 reg;
                dequantize_q4_0_t4(vb, il, reg);
                ushort d4 = b * BLK4 + il;
                local_out[d4 * 4 + 0] += weight * reg[0];
                local_out[d4 * 4 + 1] += weight * reg[1];
                local_out[d4 * 4 + 2] += weight * reg[2];
                local_out[d4 * 4 + 3] += weight * reg[3];
            }
        }
    }

    // SIMD reduction
    float sg_max = simd_max(local_max);
    float scale_factor = exp(local_max - sg_max);
    float scaled_sum = local_sum * scale_factor;
    for (ushort d = 0; d < args.head_dim; d++) {
        local_out[d] *= scale_factor;
    }

    float sg_sum = simd_sum(scaled_sum);

    // Output
    if (tiisg == 0) {
        for (ushort d = 0; d < args.head_dim; d++) {
            output[d] = local_out[d] / sg_sum;
        }
    }
}
"#;

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel = library
        .get_function("debug_attention", None)
        .expect("Kernel not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Pipeline failed");

    let q_buf = device.new_buffer(
        (q.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer(
        k_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer(
        v_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (head_dim * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        std::ptr::copy_nonoverlapping(
            q.as_ptr(),
            q_buf.contents() as *mut f32,
            q.len(),
        );
        std::ptr::copy_nonoverlapping(
            k_quant.as_ptr(),
            k_buf.contents() as *mut u8,
            k_quant.len(),
        );
        std::ptr::copy_nonoverlapping(
            v_quant.as_ptr(),
            v_buf.contents() as *mut u8,
            v_quant.len(),
        );
    }

    #[repr(C)]
    struct Args {
        seq_k: u32,
        head_dim: u32,
        scale: f32,
        k_stride_s: u32,
        v_stride_s: u32,
    }

    let row_size = (blocks_per_row * std::mem::size_of::<BlockQ4_0>()) as u32;
    let args = Args {
        seq_k: seq_k as u32,
        head_dim: head_dim as u32,
        scale,
        k_stride_s: row_size,
        v_stride_s: row_size,
    };

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
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_output = unsafe {
        std::slice::from_raw_parts(out_buf.contents() as *const f32, head_dim)
            .to_vec()
    };

    println!("CPU output [0..4]: {:?}", &cpu_output[0..4]);
    println!("GPU output [0..4]: {:?}", &gpu_output[0..4]);
    println!();

    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max diff: {:.6e}", max_diff);
    println!("Status: {}", if max_diff < 1e-3 { "PASS" } else { "FAIL" });
}

fn rng(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn cpu_attention(
    q: &[f32],
    k_quant: &[u8],
    v_quant: &[u8],
    seq_k: usize,
    head_dim: usize,
    blocks_per_row: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; seq_k];

    // Compute scores
    for sk in 0..seq_k {
        let k_blocks: Vec<BlockQ4_0> = (0..blocks_per_row)
            .map(|b| {
                let offset =
                    sk * blocks_per_row * std::mem::size_of::<BlockQ4_0>()
                        + b * std::mem::size_of::<BlockQ4_0>();
                unsafe {
                    std::ptr::read(
                        k_quant[offset..].as_ptr() as *const BlockQ4_0
                    )
                }
            })
            .collect();

        for d in 0..head_dim {
            scores[sk] += q[d] * dequantize_q4_0_value(&k_blocks, d);
        }
        scores[sk] *= scale;
    }

    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();

    // Weighted sum
    let mut output = vec![0.0f32; head_dim];
    for sk in 0..seq_k {
        let weight = (scores[sk] - max_score).exp() / exp_sum;

        let v_blocks: Vec<BlockQ4_0> = (0..blocks_per_row)
            .map(|b| {
                let offset =
                    sk * blocks_per_row * std::mem::size_of::<BlockQ4_0>()
                        + b * std::mem::size_of::<BlockQ4_0>();
                unsafe {
                    std::ptr::read(
                        v_quant[offset..].as_ptr() as *const BlockQ4_0
                    )
                }
            })
            .collect();

        for d in 0..head_dim {
            output[d] += weight * dequantize_q4_0_value(&v_blocks, d);
        }
    }

    output
}

fn dequantize_q4_0_value(blocks: &[BlockQ4_0], idx: usize) -> f32 {
    let block_idx = idx / 32;
    let elem_idx = idx % 32;
    let block = &blocks[block_idx];
    let d = half::f16::from_bits(block.d).to_f32();

    let il = elem_idx / 4;
    let sub_idx = elem_idx % 4;

    let qs_u16 = unsafe {
        std::slice::from_raw_parts(block.qs.as_ptr() as *const u16, 8)
    };

    let d1 = if il >= 4 { d / 16.0 } else { d };
    let d2 = d1 / 256.0;
    let md = -8.0 * d;
    let mask0: u16 = if il >= 4 { 0x00F0 } else { 0x000F };
    let mask1 = mask0 << 8;

    let i = if sub_idx < 2 { 0 } else { 1 };
    let q_val = qs_u16[2 * (il % 4) + i];

    if sub_idx % 2 == 0 {
        d1 * (q_val & mask0) as f32 + md
    } else {
        d2 * (q_val & mask1) as f32 + md
    }
}
