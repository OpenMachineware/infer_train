// Correctness test for quantized flash attention kernels
// Compares GPU output against CPU reference implementation

use infer_train::quant::kv_cache::{
    quantize_row_iq4_nl, quantize_row_q2_k_kv, quantize_row_q3_k_kv,
    quantize_row_q4_0, quantize_row_q4_1, quantize_row_q4_k_kv,
    quantize_row_q5_0, quantize_row_q5_1, quantize_row_q5_k_kv,
    quantize_row_q6_k_kv, quantize_row_q8_0, BlockIQ4NL, BlockQ2K_KV,
    BlockQ3K_KV, BlockQ4K_KV, BlockQ4_0, BlockQ4_1, BlockQ5K_KV, BlockQ5_0,
    BlockQ5_1, BlockQ6K_KV, BlockQ8_0,
};
use metal::*;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("=== Quantized Flash Attention Correctness Test ===\n");

    // Test configurations
    let configs = [
        ("Decode 128, Q4_0", 1, 1, 1, 128, 128, "q4_0"),
        ("Decode 256, Q4_0", 1, 1, 1, 256, 128, "q4_0"),
        ("Decode 128, Q4_1", 1, 1, 1, 128, 128, "q4_1"),
        ("Decode 128, Q5_0", 1, 1, 1, 128, 128, "q5_0"),
        ("Decode 128, Q5_1", 1, 1, 1, 128, 128, "q5_1"),
        ("Decode 128, Q8_0", 1, 1, 1, 128, 128, "q8_0"),
        ("Decode 128, F16", 1, 1, 1, 128, 128, "f16"),
        ("Decode 128, BF16", 1, 1, 1, 128, 128, "bf16"),
        ("Decode 128, IQ4_NL", 1, 1, 1, 128, 128, "iq4_nl"),
        ("Decode 128, Q2_K", 1, 1, 1, 128, 128, "q2_K"),
        ("Decode 128, Q3_K", 1, 1, 1, 128, 128, "q3_K"),
        ("Decode 128, Q4_K", 1, 1, 1, 128, 128, "q4_K"),
        ("Decode 128, Q5_K", 1, 1, 1, 128, 128, "q5_K"),
        ("Decode 128, Q6_K", 1, 1, 1, 128, 128, "q6_K"),
        ("Prefill 4, Q4_0", 1, 1, 4, 128, 128, "q4_0"),
        ("Prefill 4, Q8_0", 1, 1, 4, 128, 128, "q8_0"),
        ("Prefill 4, F16", 1, 1, 4, 128, 128, "f16"),
        ("Prefill 4, BF16", 1, 1, 4, 128, 128, "bf16"),
        ("Prefill 4, IQ4_NL", 1, 1, 4, 128, 128, "iq4_nl"),
        ("Prefill 4, Q2_K", 1, 1, 4, 128, 128, "q2_K"),
        ("Prefill 4, Q4_K", 1, 1, 4, 128, 128, "q4_K"),
    ];

    let mut seed: u64 = 12345;

    for (name, batch, heads, seq_q, seq_k, head_dim, format) in configs {
        println!("Testing: {}", name);
        println!(
            "  batch={}, heads={}, seq_q={}, seq_k={}, head_dim={}, format={}",
            batch, heads, seq_q, seq_k, head_dim, format
        );

        // Create random data
        let q_size = batch * heads * seq_q * head_dim;
        let k_size = batch * heads * seq_k * head_dim;
        let v_size = batch * heads * seq_k * head_dim;

        let q_data: Vec<f32> = (0..q_size).map(|_| rng(&mut seed)).collect();
        let k_data: Vec<f32> = (0..k_size).map(|_| rng(&mut seed)).collect();
        let v_data: Vec<f32> = (0..v_size).map(|_| rng(&mut seed)).collect();

        // Quantize K and V
        let (k_quant, v_quant, k_scale, v_scale) = match format {
            "q4_0" => {
                let k_quant =
                    quantize_k_v_q4_0(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q4_0(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q4_1" => {
                let k_quant =
                    quantize_k_v_q4_1(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q4_1(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q5_0" => {
                let k_quant =
                    quantize_k_v_q5_0(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q5_0(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q5_1" => {
                let k_quant =
                    quantize_k_v_q5_1(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q5_1(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q8_0" => {
                let k_quant =
                    quantize_k_v_q8_0(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q8_0(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "f16" => {
                let k_quant =
                    quantize_k_v_f16(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_f16(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q2_K" => {
                let k_quant =
                    quantize_k_v_q2_k(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q2_k(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q3_K" => {
                let k_quant =
                    quantize_k_v_q3_k(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q3_k(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q4_K" => {
                let k_quant =
                    quantize_k_v_q4_k(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q4_k(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q5_K" => {
                let k_quant =
                    quantize_k_v_q5_k(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q5_k(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "q6_K" => {
                let k_quant =
                    quantize_k_v_q6_k(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_q6_k(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "bf16" => {
                let k_quant =
                    quantize_k_v_bf16(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_bf16(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            "iq4_nl" => {
                let k_quant =
                    quantize_k_v_iq4_nl(&k_data, batch, heads, seq_k, head_dim);
                let v_quant =
                    quantize_k_v_iq4_nl(&v_data, batch, heads, seq_k, head_dim);
                (k_quant, v_quant, 1.0, 1.0)
            }
            _ => panic!("Unknown format: {}", format),
        };

        // CPU reference with quantized dequantization
        let cpu_output = cpu_attention_quant(
            &q_data, &k_quant, &v_quant, batch, heads, seq_q, seq_k, head_dim,
            format,
        );

        // GPU implementation
        let gpu_output = run_gpu_flash_attention_quant(
            &device, &queue, &q_data, &k_quant, &v_quant, batch, heads, seq_q,
            seq_k, head_dim, format,
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
        println!("  Status: {}", if max_diff < 1e-3 { "PASS" } else { "FAIL" });
        println!();
    }
}

fn rng(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn quantize_k_v_q4_0(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ4_0>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                // Create uninitialized blocks
                let mut blocks: Vec<BlockQ4_0> =
                    Vec::with_capacity(blocks_per_row);
                unsafe {
                    blocks.set_len(blocks_per_row);
                }
                quantize_row_q4_0(row, &mut blocks);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_q8_0(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ8_0>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                // Create uninitialized blocks
                let mut blocks: Vec<BlockQ8_0> =
                    Vec::with_capacity(blocks_per_row);
                unsafe {
                    blocks.set_len(blocks_per_row);
                }
                quantize_row_q8_0(row, &mut blocks);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn dequantize_q4_0(block: &BlockQ4_0) -> [f32; 32] {
    let d = half::f16::from_bits(block.d).to_f32();
    let mut output = [0.0f32; 32];

    // Match Metal's interleaved layout (from q4_0_layout_test.rs verification)
    // Metal reads qs as uint16_t pairs and processes in a specific pattern
    let qs_u16 = unsafe {
        std::slice::from_raw_parts(block.qs.as_ptr() as *const u16, 8)
    };

    for il in 0..8 {
        let d1 = if il >= 4 { d / 16.0 } else { d };
        let d2 = d1 / 256.0;
        let md = -8.0 * d;
        let mask0: u16 = if il >= 4 { 0x00F0 } else { 0x000F };
        let mask1 = mask0 << 8;

        for i in 0..2 {
            let q_val = qs_u16[2 * (il % 4) + i];
            let idx = il * 4 + 2 * i;
            output[idx] = d1 * (q_val & mask0) as f32 + md;
            output[idx + 1] = d2 * (q_val & mask1) as f32 + md;
        }
    }

    output
}

fn dequantize_q8_0(block: &BlockQ8_0) -> [f32; 32] {
    let d = half::f16::from_bits(block.d).to_f32();
    let mut output = [0.0f32; 32];

    // Match Metal's layout: reads as packed_char4
    for il in 0..8 {
        let qs_i32: [i32; 4] = [
            block.qs[il * 4] as i32,
            block.qs[il * 4 + 1] as i32,
            block.qs[il * 4 + 2] as i32,
            block.qs[il * 4 + 3] as i32,
        ];
        for j in 0..4 {
            output[il * 4 + j] = d * qs_i32[j] as f32;
        }
    }

    output
}

fn quantize_k_v_q4_1(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ4_1>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                let mut blocks: Vec<BlockQ4_1> =
                    Vec::with_capacity(blocks_per_row);
                unsafe {
                    blocks.set_len(blocks_per_row);
                }
                quantize_row_q4_1(row, &mut blocks);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_q5_0(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ5_0>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                let mut blocks: Vec<BlockQ5_0> =
                    Vec::with_capacity(blocks_per_row);
                unsafe {
                    blocks.set_len(blocks_per_row);
                }
                quantize_row_q5_0(row, &mut blocks);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_q5_1(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ5_1>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                let mut blocks: Vec<BlockQ5_1> =
                    Vec::with_capacity(blocks_per_row);
                unsafe {
                    blocks.set_len(blocks_per_row);
                }
                quantize_row_q5_1(row, &mut blocks);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_f16(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let row_size = head_dim * 2; // F16 is 2 bytes per element

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                // Convert f32 to f16
                let f16_row: Vec<u16> = row
                    .iter()
                    .map(|&f| half::f16::from_f32(f).to_bits())
                    .collect();
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        f16_row.as_ptr() as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_bf16(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let row_size = head_dim * 2; // BF16 is 2 bytes per element

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                // Convert f32 to bf16 (truncate lower 16 bits, keep upper 16 bits)
                let bf16_row: Vec<u16> = row
                    .iter()
                    .map(|&f| {
                        // BF16 is the upper 16 bits of FP32
                        let bits = f.to_bits();
                        (bits >> 16) as u16
                    })
                    .collect();
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        bf16_row.as_ptr() as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_iq4_nl(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockIQ4NL>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                let mut blocks: Vec<BlockIQ4NL> =
                    Vec::with_capacity(blocks_per_row);
                unsafe {
                    blocks.set_len(blocks_per_row);
                }
                quantize_row_iq4_nl(row, &mut blocks);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr() as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn dequantize_f16(data: &[u8], offset: usize, head_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; head_dim];
    let f16_data = unsafe {
        std::slice::from_raw_parts(
            data[offset..].as_ptr() as *const u16,
            head_dim,
        )
    };
    for i in 0..head_dim {
        output[i] = half::f16::from_bits(f16_data[i]).to_f32();
    }
    output
}

fn dequantize_bf16(data: &[u8], offset: usize, head_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; head_dim];
    let bf16_data = unsafe {
        std::slice::from_raw_parts(
            data[offset..].as_ptr() as *const u16,
            head_dim,
        )
    };
    for i in 0..head_dim {
        // BF16 is the upper 16 bits of FP32
        let bits = (bf16_data[i] as u32) << 16;
        output[i] = f32::from_bits(bits);
    }
    output
}

fn dequantize_iq4_nl(block: &BlockIQ4NL) -> [f32; 32] {
    use infer_train::quant::kv_cache::KVALUES_IQ4NL;
    let d = half::f16::from_bits(block.d).to_f32();
    let mut output = [0.0f32; 32];

    // Match Metal shader's dequantize_iq4_nl_t4 exactly
    // Metal processes 4 elements at a time via il parameter (il = 0..7)
    // Each il reads 2 bytes from qs[il*2 .. il*2+1]
    // Output is sequential: il*4, il*4+1, il*4+2, il*4+3
    // But the values are interleaved from the packing order

    // Rust quantizer packs: qs[j] = (index[j] | index[j+16] << 4)
    // So qs[0] contains index[0] (low) and index[16] (high)
    //    qs[1] contains index[1] (low) and index[17] (high)
    //    etc.

    // Metal reads for il=0: qs[0], qs[1]
    //   output[0] = kvalues[qs[0].low]  = index[0]
    //   output[1] = kvalues[qs[0].high] = index[16]
    //   output[2] = kvalues[qs[1].low]  = index[1]
    //   output[3] = kvalues[qs[1].high] = index[17]
    // This is WRONG! We need sequential output.

    // The correct approach: dequantize to sequential order
    // Output should be: [index[0], index[1], ..., index[15], index[16], ...]

    for j in 0..16 {
        let byte = block.qs[j];
        let idx_low = (byte & 0x0F) as usize;
        let idx_high = ((byte >> 4) & 0x0F) as usize;
        output[j] = KVALUES_IQ4NL[idx_low] * d; // elements 0..15
        output[j + 16] = KVALUES_IQ4NL[idx_high] * d; // elements 16..31
    }
    output
}

// K-quant quantization functions
fn quantize_k_v_q2_k(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let row_size = std::mem::size_of::<BlockQ2K_KV>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                // Pad to 256 elements if needed (for head_dim < 256)
                let mut padded = row.to_vec();
                padded.resize(256, 0.0);

                let mut block =
                    BlockQ2K_KV { d: 0, dmin: 0, scales: [0; 16], qs: [0; 64] };
                quantize_row_q2_k_kv(&padded, &mut block);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_q3_k(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let row_size = std::mem::size_of::<BlockQ3K_KV>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                let mut padded = row.to_vec();
                padded.resize(256, 0.0);

                let mut block = BlockQ3K_KV {
                    d: 0,
                    hmask: [0; 32],
                    qs: [0; 64],
                    scales: [0; 12],
                };
                quantize_row_q3_k_kv(&padded, &mut block);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_q4_k(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let row_size = std::mem::size_of::<BlockQ4K_KV>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                let mut padded = row.to_vec();
                padded.resize(256, 0.0);

                let mut block = BlockQ4K_KV {
                    d: 0,
                    dmin: 0,
                    scales: [0; 12],
                    qs: [0; 128],
                };
                quantize_row_q4_k_kv(&padded, &mut block);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_q5_k(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let row_size = std::mem::size_of::<BlockQ5K_KV>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                let mut padded = row.to_vec();
                padded.resize(256, 0.0);

                let mut block = BlockQ5K_KV {
                    d: 0,
                    dmin: 0,
                    scales: [0; 12],
                    qh: [0; 32],
                    qs: [0; 128],
                };
                quantize_row_q5_k_kv(&padded, &mut block);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

fn quantize_k_v_q6_k(
    data: &[f32],
    batch: usize,
    heads: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<u8> {
    let mut output = Vec::new();
    let row_size = std::mem::size_of::<BlockQ6K_KV>();

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_k {
                let offset = b * heads * seq_k * head_dim
                    + h * seq_k * head_dim
                    + s * head_dim;
                let row = &data[offset..offset + head_dim];

                let mut padded = row.to_vec();
                padded.resize(256, 0.0);

                let mut block = BlockQ6K_KV {
                    d: 0,
                    scales: [0; 16],
                    ql: [0; 128],
                    qh: [0; 64],
                };
                quantize_row_q6_k_kv(&padded, &mut block);

                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &block as *const _ as *const u8,
                        row_size,
                    )
                };
                output.extend_from_slice(bytes);
            }
        }
    }
    output
}

// K-quant dequantization functions
fn dequantize_q2_k(block: &BlockQ2K_KV) -> [f32; 256] {
    let d = half::f16::from_bits(block.d).to_f32();
    let min = half::f16::from_bits(block.dmin).to_f32();
    let mut output = [0.0f32; 256];

    // Q2_K: 2-bit quantization, 4 elements per byte
    // scales[16]: one scale per 16-element sub-block
    // qs[64]: 64 bytes for 256 elements (4 2-bit values per byte)

    for j in 0..64 {
        let byte = block.qs[j];

        // Each byte has 4 2-bit values
        for k in 0..4 {
            let idx = j * 4 + k;
            let val = ((byte >> (2 * k)) & 0x03) as f32;

            // Scale from sub-block
            let sub_block = idx / 16;
            let sc = block.scales[sub_block];
            let scale = (sc & 0x0F) as f32;

            output[idx] = d * scale * val - min;
        }
    }
    output
}

fn dequantize_q3_k(block: &BlockQ3K_KV) -> [f32; 256] {
    let d = half::f16::from_bits(block.d).to_f32();
    let mut output = [0.0f32; 256];

    // Q3_K: 3-bit quantization
    // qs[64]: low 2 bits, 4 elements per byte
    // hmask[32]: high bit for each element
    // scales[12]: scale per sub-block

    for j in 0..64 {
        let byte = block.qs[j];

        for k in 0..4 {
            let idx = j * 4 + k;
            let low2 = ((byte >> (2 * k)) & 0x03) as f32;
            let high1 = ((block.hmask[idx / 8] >> (idx % 8)) & 1) as f32;

            // 3-bit value: low2 + high1 * 4
            let val = low2 + high1 * 4.0 - 4.0; // Signed: [-4, 3]

            // Scale
            let sub_block = idx / 32;
            let sc = block.scales[sub_block] as f32;

            output[idx] = d * sc * val;
        }
    }
    output
}

fn dequantize_q4_k(block: &BlockQ4K_KV) -> [f32; 256] {
    let d = half::f16::from_bits(block.d).to_f32();
    let dmin = half::f16::from_bits(block.dmin).to_f32();
    let mut output = [0.0f32; 256];

    // qs layout: each byte stores two 4-bit values
    // qs[j] = element j (low nibble) | element j+128 (high nibble)
    // For head_dim=128, we use elements 0..127 (low nibbles from qs[0..127])
    // Quantization: v0 = (x[j] * id - dmin / d) → x = v * d + dmin

    for j in 0..128 {
        let byte = block.qs[j];
        let low = (byte & 0x0F) as f32;
        let high = ((byte >> 4) & 0x0F) as f32;

        output[j] = low * d + dmin;
        output[j + 128] = high * d + dmin;
    }
    output
}

fn dequantize_q5_k(block: &BlockQ5K_KV) -> [f32; 256] {
    let d = half::f16::from_bits(block.d).to_f32();
    let dmin = half::f16::from_bits(block.dmin).to_f32();
    let mut output = [0.0f32; 256];

    // qs layout: each byte stores two 4-bit values (low 4 bits of 5-bit value)
    // qh stores high bit for each element
    // Quantization: v0 = (x[j] * id - dmin / d).clamp(0.0, 31.0) → x = v * d + dmin
    for j in 0..128 {
        let byte = block.qs[j];
        let low = (byte & 0x0F) as f32;
        let high = ((byte >> 4) & 0x0F) as f32;

        // High bit from qh
        let h0 = ((block.qh[j / 8] >> (j % 8)) & 1) as f32;
        let h1 = ((block.qh[(j + 128) / 8] >> ((j + 128) % 8)) & 1) as f32;

        let val0 = low + h0 * 16.0;
        let val1 = high + h1 * 16.0;

        output[j] = val0 * d + dmin;
        output[j + 128] = val1 * d + dmin;
    }
    output
}

fn dequantize_q6_k(block: &BlockQ6K_KV) -> [f32; 256] {
    let d = half::f16::from_bits(block.d).to_f32();
    let mut output = [0.0f32; 256];

    // Q6_K: 6-bit signed quantization
    // ql[128]: lower 4 bits, two elements per byte
    // qh[64]: upper 2 bits, packed in specific layout
    // Quantization: v0 = (x[j] * id + 32.0).clamp(0.0, 63.0)
    // Dequant: x = v0 * d - 32.0

    for j in 0..128 {
        let ql_byte = block.ql[j];
        let low4 = (ql_byte & 0x0F) as f32;
        let high4 = ((ql_byte >> 4) & 0x0F) as f32;

        // Upper 2 bits from qh
        let qh_idx = j / 2;
        let offset = (j % 2) * 4;
        let qh_byte = block.qh[qh_idx];
        let h0 = ((qh_byte >> offset) & 0x03) as f32;
        let h1 = ((qh_byte >> (offset + 2)) & 0x03) as f32;

        // Reconstruct 6-bit value: lower 4 bits + upper 2 bits * 16
        let val0 = low4 + h0 * 16.0 - 32.0;
        let val1 = high4 + h1 * 16.0 - 32.0;

        output[j] = val0 * d;
        output[j + 128] = val1 * d;
    }
    output
}

fn dequantize_q4_1(block: &BlockQ4_1) -> [f32; 32] {
    let d = half::f16::from_bits(unsafe { std::ptr::addr_of!(block.d).read() })
        .to_f32();
    let m = half::f16::from_bits(unsafe { std::ptr::addr_of!(block.m).read() })
        .to_f32();
    let mut output = [0.0f32; 32];

    let qs_u16 = unsafe {
        std::slice::from_raw_parts(block.qs.as_ptr() as *const u16, 8)
    };

    for il in 0..8 {
        let d1 = if il >= 4 { d / 16.0 } else { d };
        let d2 = d1 / 256.0;
        let mask0: u16 = if il >= 4 { 0x00F0 } else { 0x000F };
        let mask1 = mask0 << 8;

        for i in 0..2 {
            let q_val = qs_u16[2 * (il % 4) + i];
            let idx = il * 4 + 2 * i;
            output[idx] = d1 * (q_val & mask0) as f32 + m;
            output[idx + 1] = d2 * (q_val & mask1) as f32 + m;
        }
    }

    output
}

fn dequantize_q5_0(block: &BlockQ5_0) -> [f32; 32] {
    let d = half::f16::from_bits(unsafe { std::ptr::addr_of!(block.d).read() })
        .to_f32();
    let md = -16.0 * d;
    let mut output = [0.0f32; 32];

    let qs_u16 = unsafe {
        std::slice::from_raw_parts(block.qs.as_ptr() as *const u16, 8)
    };
    let qh = u32::from_le_bytes(block.qh);

    for il in 0..8 {
        let mask: u16 = if il >= 4 { 0x00F0 } else { 0x000F };
        let x_mv = if il >= 4 { 4 } else { 0 };
        let gh_mv = if il >= 4 { 12 } else { 0 };
        let gh_bk = if il >= 4 { 0 } else { 4 };

        for ii in 0..2 {
            let i = 2 * (il % 4) + ii;
            let xh_0 = ((qh >> (gh_mv + 2 * i)) << gh_bk) as u8 & 0x10;
            let xh_1 = ((qh >> (gh_mv + 2 * i + 1)) << gh_bk) as u8 & 0x10;

            let x0 = ((qs_u16[i] & mask) >> x_mv) as i32 | xh_0 as i32;
            let x1 = (((qs_u16[i] >> 8) & mask) >> x_mv) as i32 | xh_1 as i32;

            let idx = il * 4 + 2 * ii;
            output[idx] = d * x0 as f32 + md;
            output[idx + 1] = d * x1 as f32 + md;
        }
    }

    output
}

fn dequantize_q5_1(block: &BlockQ5_1) -> [f32; 32] {
    let d = half::f16::from_bits(unsafe { std::ptr::addr_of!(block.d).read() })
        .to_f32();
    let m = half::f16::from_bits(unsafe { std::ptr::addr_of!(block.m).read() })
        .to_f32();
    let mut output = [0.0f32; 32];

    let qs_u16 = unsafe {
        std::slice::from_raw_parts(block.qs.as_ptr() as *const u16, 8)
    };
    let qh = u32::from_le_bytes(block.qh);

    for il in 0..8 {
        let mask: u16 = if il >= 4 { 0x00F0 } else { 0x000F };
        let x_mv = if il >= 4 { 4 } else { 0 };
        let gh_mv = if il >= 4 { 12 } else { 0 };
        let gh_bk = if il >= 4 { 0 } else { 4 };

        for ii in 0..2 {
            let i = 2 * (il % 4) + ii;
            let xh_0 = ((qh >> (gh_mv + 2 * i)) << gh_bk) as u8 & 0x10;
            let xh_1 = ((qh >> (gh_mv + 2 * i + 1)) << gh_bk) as u8 & 0x10;

            let x0 = ((qs_u16[i] & mask) >> x_mv) as i32 | xh_0 as i32;
            let x1 = (((qs_u16[i] >> 8) & mask) >> x_mv) as i32 | xh_1 as i32;

            let idx = il * 4 + 2 * ii;
            output[idx] = d * x0 as f32 + m;
            output[idx + 1] = d * x1 as f32 + m;
        }
    }

    output
}

fn cpu_attention_quant(
    q: &[f32],
    k_quant: &[u8],
    v_quant: &[u8],
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    format: &str,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * heads * seq_q * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    let is_kquant = format.starts_with("q")
        && format.len() > 3
        && !format.contains("_0")
        && !format.contains("_1");
    let is_linear = format == "f16" || format == "bf16";
    let blocks_per_row = if is_linear || is_kquant { 0 } else { head_dim / 32 };
    let row_size = match format {
        "q4_0" => blocks_per_row * std::mem::size_of::<BlockQ4_0>(),
        "q4_1" => blocks_per_row * std::mem::size_of::<BlockQ4_1>(),
        "q5_0" => blocks_per_row * std::mem::size_of::<BlockQ5_0>(),
        "q5_1" => blocks_per_row * std::mem::size_of::<BlockQ5_1>(),
        "q8_0" => blocks_per_row * std::mem::size_of::<BlockQ8_0>(),
        "f16" => head_dim * 2,
        "bf16" => head_dim * 2,
        "iq4_nl" => (head_dim / 32) * std::mem::size_of::<BlockIQ4NL>(),
        "q2_K" => std::mem::size_of::<BlockQ2K_KV>(),
        "q3_K" => std::mem::size_of::<BlockQ3K_KV>(),
        "q4_K" => std::mem::size_of::<BlockQ4K_KV>(),
        "q5_K" => std::mem::size_of::<BlockQ5K_KV>(),
        "q6_K" => std::mem::size_of::<BlockQ6K_KV>(),
        _ => panic!("Unknown format"),
    };

    for b in 0..batch {
        for h in 0..heads {
            for sq in 0..seq_q {
                let q_offset = b * heads * seq_q * head_dim
                    + h * seq_q * head_dim
                    + sq * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                let mut scores = vec![0.0f32; seq_k];
                for sk in 0..seq_k {
                    // Dequantize K
                    let k_row_byte_offset =
                        (b * heads * seq_k + h * seq_k + sk) * row_size;
                    let k_deq = if format == "f16" {
                        dequantize_f16(k_quant, k_row_byte_offset, head_dim)
                    } else if format == "bf16" {
                        dequantize_bf16(k_quant, k_row_byte_offset, head_dim)
                    } else if is_kquant {
                        // K-quant: 256 elements per block, we need only head_dim
                        let block_bytes = &k_quant[k_row_byte_offset..];
                        let full_deq: [f32; 256] = match format {
                            "q2_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ2K_KV)
                                };
                                dequantize_q2_k(block)
                            }
                            "q3_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ3K_KV)
                                };
                                dequantize_q3_k(block)
                            }
                            "q4_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ4K_KV)
                                };
                                dequantize_q4_k(block)
                            }
                            "q5_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ5K_KV)
                                };
                                dequantize_q5_k(block)
                            }
                            "q6_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ6K_KV)
                                };
                                dequantize_q6_k(block)
                            }
                            _ => panic!("Unknown K-quant format"),
                        };
                        full_deq[..head_dim].to_vec()
                    } else if format == "iq4_nl" {
                        let mut k_deq = vec![0.0f32; head_dim];
                        for blk in 0..blocks_per_row {
                            let block_bytes = &k_quant[k_row_byte_offset
                                + blk * std::mem::size_of::<BlockIQ4NL>()..];
                            let block = unsafe {
                                &*(block_bytes.as_ptr() as *const BlockIQ4NL)
                            };
                            let deq = dequantize_iq4_nl(block);
                            for i in 0..32 {
                                k_deq[blk * 32 + i] = deq[i];
                            }
                        }
                        k_deq
                    } else {
                        let mut k_deq = vec![0.0f32; head_dim];
                        for blk in 0..blocks_per_row {
                            let deq = match format {
                                "q4_0" => {
                                    let block_bytes = &k_quant[k_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ4_0>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ4_0)
                                    };
                                    dequantize_q4_0(block)
                                }
                                "q4_1" => {
                                    let block_bytes = &k_quant[k_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ4_1>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ4_1)
                                    };
                                    dequantize_q4_1(block)
                                }
                                "q5_0" => {
                                    let block_bytes = &k_quant[k_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ5_0>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ5_0)
                                    };
                                    dequantize_q5_0(block)
                                }
                                "q5_1" => {
                                    let block_bytes = &k_quant[k_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ5_1>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ5_1)
                                    };
                                    dequantize_q5_1(block)
                                }
                                "q8_0" => {
                                    let block_bytes = &k_quant[k_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ8_0>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ8_0)
                                    };
                                    dequantize_q8_0(block)
                                }
                                _ => panic!("Unknown format"),
                            };

                            for i in 0..32 {
                                k_deq[blk * 32 + i] = deq[i];
                            }
                        }
                        k_deq
                    };

                    // Q*K
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q_vec[d] * k_deq[d];
                    }
                    scores[sk] = score * scale;
                }

                // Softmax
                let max_score =
                    scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 =
                    scores.iter().map(|s| (s - max_score).exp()).sum();

                // Weighted sum of V
                let o_offset = b * heads * seq_q * head_dim
                    + h * seq_q * head_dim
                    + sq * head_dim;
                for sk in 0..seq_k {
                    let weight = (scores[sk] - max_score).exp() / exp_sum;

                    // Dequantize V
                    let v_row_byte_offset =
                        (b * heads * seq_k + h * seq_k + sk) * row_size;
                    let v_deq = if format == "f16" {
                        dequantize_f16(v_quant, v_row_byte_offset, head_dim)
                    } else if format == "bf16" {
                        dequantize_bf16(v_quant, v_row_byte_offset, head_dim)
                    } else if is_kquant {
                        let block_bytes = &v_quant[v_row_byte_offset..];
                        let full_deq: [f32; 256] = match format {
                            "q2_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ2K_KV)
                                };
                                dequantize_q2_k(block)
                            }
                            "q3_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ3K_KV)
                                };
                                dequantize_q3_k(block)
                            }
                            "q4_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ4K_KV)
                                };
                                dequantize_q4_k(block)
                            }
                            "q5_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ5K_KV)
                                };
                                dequantize_q5_k(block)
                            }
                            "q6_K" => {
                                let block = unsafe {
                                    &*(block_bytes.as_ptr()
                                        as *const BlockQ6K_KV)
                                };
                                dequantize_q6_k(block)
                            }
                            _ => panic!("Unknown K-quant format"),
                        };
                        full_deq[..head_dim].to_vec()
                    } else if format == "iq4_nl" {
                        let mut v_deq = vec![0.0f32; head_dim];
                        for blk in 0..blocks_per_row {
                            let block_bytes = &v_quant[v_row_byte_offset
                                + blk * std::mem::size_of::<BlockIQ4NL>()..];
                            let block = unsafe {
                                &*(block_bytes.as_ptr() as *const BlockIQ4NL)
                            };
                            let deq = dequantize_iq4_nl(block);
                            for i in 0..32 {
                                v_deq[blk * 32 + i] = deq[i];
                            }
                        }
                        v_deq
                    } else {
                        let mut v_deq = vec![0.0f32; head_dim];
                        for blk in 0..blocks_per_row {
                            let deq = match format {
                                "q4_0" => {
                                    let block_bytes = &v_quant[v_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ4_0>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ4_0)
                                    };
                                    dequantize_q4_0(block)
                                }
                                "q4_1" => {
                                    let block_bytes = &v_quant[v_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ4_1>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ4_1)
                                    };
                                    dequantize_q4_1(block)
                                }
                                "q5_0" => {
                                    let block_bytes = &v_quant[v_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ5_0>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ5_0)
                                    };
                                    dequantize_q5_0(block)
                                }
                                "q5_1" => {
                                    let block_bytes = &v_quant[v_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ5_1>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ5_1)
                                    };
                                    dequantize_q5_1(block)
                                }
                                "q8_0" => {
                                    let block_bytes = &v_quant[v_row_byte_offset
                                        + blk
                                            * std::mem::size_of::<BlockQ8_0>(
                                            )..];
                                    let block = unsafe {
                                        &*(block_bytes.as_ptr()
                                            as *const BlockQ8_0)
                                    };
                                    dequantize_q8_0(block)
                                }
                                _ => panic!("Unknown format"),
                            };

                            for i in 0..32 {
                                v_deq[blk * 32 + i] = deq[i];
                            }
                        }
                        v_deq
                    };

                    for d in 0..head_dim {
                        output[o_offset + d] += weight * v_deq[d];
                    }
                }
            }
        }
    }

    output
}

fn run_gpu_flash_attention_quant(
    device: &Device,
    queue: &CommandQueue,
    q: &[f32],
    k_quant: &[u8],
    v_quant: &[u8],
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    format: &str,
) -> Vec<f32> {
    // Compile kernel
    let source = include_str!("../../shaders/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");

    // Choose kernel
    let kernel_name = if seq_q == 1 {
        format!("flash_decode_attention_{}", format)
    } else {
        format!("flash_prefill_attention_{}", format)
    };

    let kernel =
        library.get_function(&kernel_name, None).expect("Kernel not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Pipeline failed");

    // Create buffers
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

    // Calculate strides for quantized data
    let blocks_per_row =
        if format == "f16" || format == "bf16" || format == "iq4_nl" {
            0
        } else {
            head_dim / 32
        };
    let k_row_size = match format {
        "q4_0" => blocks_per_row * std::mem::size_of::<BlockQ4_0>(),
        "q4_1" => blocks_per_row * std::mem::size_of::<BlockQ4_1>(),
        "q5_0" => blocks_per_row * std::mem::size_of::<BlockQ5_0>(),
        "q5_1" => blocks_per_row * std::mem::size_of::<BlockQ5_1>(),
        "q8_0" => blocks_per_row * std::mem::size_of::<BlockQ8_0>(),
        "f16" => head_dim * 2,
        "bf16" => head_dim * 2,
        "iq4_nl" => (head_dim / 32) * std::mem::size_of::<BlockIQ4NL>(),
        // K-quant formats: one 256-element block per head_dim
        "q2_K" => std::mem::size_of::<BlockQ2K_KV>(),
        "q3_K" => std::mem::size_of::<BlockQ3K_KV>(),
        "q4_K" => std::mem::size_of::<BlockQ4K_KV>(),
        "q5_K" => std::mem::size_of::<BlockQ5K_KV>(),
        "q6_K" => std::mem::size_of::<BlockQ6K_KV>(),
        _ => panic!("Unknown format"),
    };

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
        k_stride_b: (heads * seq_k * k_row_size) as u32,
        k_stride_h: (seq_k * k_row_size) as u32,
        k_stride_s: k_row_size as u32,
        v_stride_b: (heads * seq_k * k_row_size) as u32,
        v_stride_h: (seq_k * k_row_size) as u32,
        v_stride_s: k_row_size as u32,
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

    let grid_size = if seq_q == 1 {
        MTLSize { width: 1, height: heads as u64, depth: batch as u64 }
    } else {
        MTLSize {
            width: seq_q as u64,
            height: heads as u64,
            depth: batch as u64,
        }
    };
    let threadgroup_size = if seq_q == 1 {
        // Decode: one SIMD group (32 threads) per head
        MTLSize { width: 32, height: 1, depth: 1 }
    } else {
        // Prefill: 128 threads for parallel KV processing
        MTLSize { width: 128, height: 1, depth: 1 }
    };
    enc.dispatch_thread_groups(grid_size, threadgroup_size);

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
