// Test Q6_K vec_dot correctness

use infer_train::types::*;
use infer_train::backend::arm::vec_dot;

fn dequant_q6_k_block(block: &BlockQ6_K) -> Vec<f32> {
    let d = half::f16::from_bits(block.d).to_f32();
    
    let mut result = vec![0.0f32; 256];
    
    // Create aux8 array with values in range -32 to 31
    let mut aux8 = [0i8; 256];
    let q4 = &block.ql;
    let qh = &block.qh;
    
    for j in (0..256).step_by(128) {
        let jj = j / 2;  // q4 advances by 64 for each 128 output elements
        for l in 0..32 {
            aux8[j + l + 0] = ((q4[jj + l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8 - 32;
            aux8[j + l + 32] = ((q4[jj + l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            aux8[j + l + 64] = ((q4[jj + l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            aux8[j + l + 96] = ((q4[jj + l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
        }
    }
    
    let scales = &block.scales;
    let mut is = 0;
    for j in 0..16 {
        let scale = scales[is] as f32;
        for l in 0..16 {
            result[j * 16 + l] = d * scale * aux8[j * 16 + l] as f32;
        }
        is += 1;
    }
    
    result
}

fn main() {
    // Create a test Q6_K block with known values
    let mut block_q6 = BlockQ6_K {
        d: half::f16::to_bits(half::f16::from_f32(0.1)),
        ql: [0; 128],
        qh: [0; 64],
        scales: [8; 16],
    };
    
    // Set ql to simple values (each byte has two 4-bit values)
    for i in 0..128 {
        block_q6.ql[i] = 0x44; // Each nibble = 4
    }
    
    // qh bits determine upper 2 bits
    for i in 0..64 {
        block_q6.qh[i] = 0x00; // Upper 2 bits = 0
    }
    
    // Create a simple Q8_K input (all ones)
    let mut block_q8 = BlockQ8_K {
        d: 1.0,
        qs: [1; 256],
        bsums: [0; 16],
    };
    for j in 0..16 {
        block_q8.bsums[j] = 16;
    }
    
    // Compute vec_dot
    let result = vec_dot::vec_dot_q6_k_q8_k(&[block_q6], &[block_q8]);
    println!("vec_dot result: {}", result);
    
    // Compare with FP32 dequantization
    let dequant = dequant_q6_k_block(&block_q6);
    let mut fp32_sum = 0.0f32;
    for i in 0..256 {
        fp32_sum += dequant[i] * 1.0; // input is all ones
    }
    println!("FP32 dequant sum: {}", fp32_sum);
    
    // Print first 16 dequant values
    println!("First 16 dequant values: {:?}", &dequant[..16]);
    
    // Expected: each aux8 value should be 4 - 32 = -28
    // scale = 8, d = 0.1
    // result = d * scale * aux8 = 0.1 * 8 * -28 = -22.4
    // sum over 256 elements = -22.4 * 256 = -5734.4
    println!("Expected aux8 value: 4 - 32 = -28");
    println!("Expected dequant per element: 0.1 * 8 * -28 = -22.4");
    println!("Expected sum: -22.4 * 256 = {}", -22.4 * 256.0);
}