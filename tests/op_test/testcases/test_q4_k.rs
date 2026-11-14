// Test Q4_K vec_dot correctness

use infer_train::types::*;
use infer_train::backend::arm::{vec_dot, quant};

fn dequant_q4_k_block(block: &BlockQ4_K) -> Vec<f32> {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    
    let mut utmp: [u32; 4] = [0; 4];
    utmp[0..3].copy_from_slice(unsafe {
        std::slice::from_raw_parts(block.scales.as_ptr() as *const u32, 3)
    });
    
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
    
    let d = half::f16::from_bits(block.d).to_f32();
    let dmin = half::f16::from_bits(block.dmin).to_f32();
    
    let mut result = vec![0.0f32; 256];
    
    // Dequantize the 4-bit values
    let mut aux8 = [0i8; 256];
    for j in 0..4 {
        for l in 0..32 {
            aux8[j * 64 + l] = (block.qs[j * 32 + l] & 0xF) as i8;
        }
        for l in 0..32 {
            aux8[j * 64 + 32 + l] = (block.qs[j * 32 + l] >> 4) as i8;
        }
    }
    
    // Apply scales and d
    let mut is = 0;
    for j in 0..8 {
        let scale = scales[is] as f32;
        for l in 0..32 {
            result[j * 32 + l] = d * scale * aux8[j * 32 + l] as f32;
        }
        is += 1;
    }
    
    // Apply mins
    for j in 0..16 {
        let min_val = mins[j / 2] as f32;
        for l in 0..16 {
            result[j * 16 + l] -= dmin * min_val;
        }
    }
    
    result
}

fn main() {
    // Create a test Q4_K block with known values
    let mut block_q4 = BlockQ4_K {
        d: half::f16::to_bits(half::f16::from_f32(0.1)),
        dmin: half::f16::to_bits(half::f16::from_f32(0.0)),
        scales: [0; 12],
        qs: [0; 128],
    };
    
    // Set scales to 8 (after 6-bit encoding)
    // We need scales[0..8] to be 8
    // scales are stored in 12 bytes, need to decode properly
    // For simplicity, let's try to set all bits to get scale = 32
    block_q4.scales[0] = 32; // This should give scales[0] = 32 after masking
    block_q4.scales[1] = 0;
    block_q4.scales[2] = 0;
    
    // Set qs to simple values
    for i in 0..128 {
        block_q4.qs[i] = 0x88; // Each nibble = 8
    }
    
    // Create a simple Q8_K input (all ones)
    let mut block_q8 = BlockQ8_K {
        d: 1.0,
        qs: [1; 256],
        bsums: [0; 16], // Each block of 16 ones sums to 16
    };
    for j in 0..16 {
        block_q8.bsums[j] = 16;
    }
    
    // Compute vec_dot
    let result = vec_dot::vec_dot_q4_k_q8_k(&[block_q4], &[block_q8]);
    println!("vec_dot result: {}", result);
    
    // Compare with FP32 dequantization
    let dequant = dequant_q4_k_block(&block_q4);
    let mut fp32_sum = 0.0f32;
    for i in 0..256 {
        fp32_sum += dequant[i] * 1.0; // input is all ones
    }
    println!("FP32 dequant sum: {}", fp32_sum);
    
    // Print first 16 dequant values
    println!("First 16 dequant values: {:?}", &dequant[..16]);
    
    // Check scales
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    
    let mut utmp: [u32; 4] = [0; 4];
    utmp[0..3].copy_from_slice(unsafe {
        std::slice::from_raw_parts(block_q4.scales.as_ptr() as *const u32, 3)
    });
    
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
    
    println!("Decoded scales: {:?}", scales);
    println!("Decoded mins: {:?}", mins);
    println!("d: {}", half::f16::from_bits(block_q4.d).to_f32());
    println!("dmin: {}", half::f16::from_bits(block_q4.dmin).to_f32());
}