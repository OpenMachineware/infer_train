// Test actual GEMV computation for L8 Q projection
use infer_train::model::GgufFile;
use infer_train::types::*;
use infer_train::backend::arm::{vec_dot, quant};
use half::f16;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");
    
    // Get L8 Q weights
    let q_weight_name = "blk.8.attn_q.weight";
    let tensor = gguf.tensors.iter().find(|t| t.name == q_weight_name).expect("Tensor not found");
    let weight_data = gguf.get_tensor_data(&tensor.name).expect("Weight data not found");
    
    let blocks_per_row = 4; // 1024 / 256
    let weights: &[BlockQ4_K] = unsafe {
        std::slice::from_raw_parts(weight_data.as_ptr() as *const BlockQ4_K, 2048 * 4)
    };
    
    // Create a simple input (all ones)
    let input: Vec<f32> = vec![1.0; 1024];
    
    // Quantize input to Q8_K
    let nb = (input.len() + QK_K - 1) / QK_K;
    let mut input_q8k: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quant::quantize_q8_k(&input, &mut input_q8k);
    
    println!("Input Q8_K blocks:");
    for (i, block) in input_q8k.iter().enumerate().take(4) {
        println!("  Block {}: d={:.6} qs[0..8]={:?}", i, block.d, &block.qs[..8]);
    }
    
    // Compute first few outputs
    println!("\nComputing first 5 outputs:");
    for i in 0..5 {
        let row_blocks = &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
        let result = vec_dot::vec_dot_q4_k_q8_k(row_blocks, &input_q8k);
        
        // Also compute using full FP32 (dequantize Q4_K weights and use original input)
        let mut fp32_sum = 0.0f32;
        let mut weight_offset = 0usize;
        for block in row_blocks.iter() {
            let d = f16::from_bits(block.d).to_f32();
            let dmin = f16::from_bits(block.dmin).to_f32();
            
            // Decode scales
            let mut utmp: [u32; 4] = [0; 4];
            utmp[0..3].copy_from_slice(unsafe {
                std::slice::from_raw_parts(block.scales.as_ptr() as *const u32, 3)
            });
            
            const KMASK1: u32 = 0x3f3f3f3f;
            const KMASK2: u32 = 0x0f0f0f0f;
            const KMASK3: u32 = 0x03030303;
            
            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;
            
            let scales: [u8; 8] = unsafe {
                std::slice::from_raw_parts(utmp[0..2].as_ptr() as *const u8, 8)
                    .try_into()
                    .unwrap()
            };
            let mins: [u8; 8] = unsafe {
                std::slice::from_raw_parts(utmp[2..4].as_ptr() as *const u8, 8)
                    .try_into()
                    .unwrap()
            };
            
            // Dequantize 4-bit values to FP32
            let mut aux8 = [0i8; 256];
            for j in 0..4 {
                for l in 0..32 {
                    aux8[j * 64 + l] = (block.qs[j * 32 + l] & 0xF) as i8;
                }
                for l in 0..32 {
                    aux8[j * 64 + 32 + l] = (block.qs[j * 32 + l] >> 4) as i8;
                }
            }
            
            let mut dequant_weight = [0.0f32; 256];
            let mut is = 0;
            for j in 0..8 {
                let scale = scales[is] as f32;
                for l in 0..32 {
                    dequant_weight[j * 32 + l] = d * scale * aux8[j * 32 + l] as f32;
                }
                is += 1;
            }
            
            // Apply mins offset
            for j in 0..16 {
                let min_val = mins[j / 2] as f32;
                for l in 0..16 {
                    dequant_weight[j * 16 + l] -= dmin * min_val;
                }
            }
            
            // Compute dot product with original FP32 input
            for l in 0..256 {
                if weight_offset + l < input.len() {
                    fp32_sum += dequant_weight[l] * input[weight_offset + l];
                }
            }
            weight_offset += 256;
        }
        
        println!("  Output[{}]: vec_dot={:.6} fp32={:.6} diff={:.6}", 
                 i, result, fp32_sum, (result - fp32_sum).abs());
    }
    
    // Now test with realistic input values (matching L7 output)
    println!("\n\nTesting with realistic input values (matching L7 output range):");
    let realistic_input: Vec<f32> = vec![30.0; 1024];  // Midpoint of L7 output range
    
    let mut input_q8k_realistic: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quant::quantize_q8_k(&realistic_input, &mut input_q8k_realistic);
    
    println!("Input Q8_K blocks (realistic):");
    for (i, block) in input_q8k_realistic.iter().enumerate().take(2) {
        println!("  Block {}: d={:.6} qs[0..8]={:?}", i, block.d, &block.qs[..8]);
    }
    
    println!("\nComputing outputs with realistic input:");
    for i in 0..5 {
        let row_blocks = &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
        let result = vec_dot::vec_dot_q4_k_q8_k(row_blocks, &input_q8k_realistic);
        println!("  Output[{}]: {:.6}", i, result);
    }
}