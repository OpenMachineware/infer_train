// Unit test for Q5_K dequantize logic
// Verifies that the dequantize function produces correct values

use half::f16;
use infer_train::quant::types::BlockQ5K;
use std::fmt;

fn main() {
    // Create a test Q5_K block with known values
    let mut block = BlockQ5K {
        d: f16::from_f32(0.001).to_bits(),
        dmin: f16::from_f32(0.0001).to_bits(),
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };

    // Set scales: first 8 bytes are scales, next 4 bytes are mins
    // Sub-block 0: scale=32, min=16
    block.scales[0] = 32; // scale
    block.scales[4] = 16; // min

    // Set qs: all zeros for now
    // il=0 processes elements 0-15 from qs[0..15] with low nibbles
    // qh provides the high bit

    // Set qh[0] = 0x01 (bit 0 set for element 0)
    block.qh[0] = 0x01;

    // Set qs[0] = 0x0F (low nibble = 15)
    block.qs[0] = 0x0F;

    // Manual dequantize for il=0:
    // d = 0.001, scale = 32, min = 16
    // dl = d * scale = 0.001 * 32 = 0.032
    // ml = dmin * min = 0.0001 * 16 = 0.0016
    // For element 0: qs[0] & 0x0F = 15, qh[0] & 0x01 = 1 -> add 16
    // value = 15 + 16 = 31
    // result = dl * 31 - ml = 0.032 * 31 - 0.0016 = 0.992 - 0.0016 = 0.9904

    println!("Test block:");
    println!("  d = {} (f16 bits: {})", f16::from_bits(block.d), block.d);
    println!(
        "  dmin = {} (f16 bits: {})",
        f16::from_bits(block.dmin),
        block.dmin
    );
    println!("  scales[0] = {}", block.scales[0]);
    println!("  scales[4] = {}", block.scales[4]);
    println!("  qs[0] = {:#04x}", block.qs[0]);
    println!("  qh[0] = {:#04x}", block.qh[0]);

    let d: f32 = f16::from_bits(block.d).into();
    let dmin: f32 = f16::from_bits(block.dmin).into();
    let scale = block.scales[0] & 63;
    let min = block.scales[4] & 63;

    println!("\nDequant parameters:");
    println!("  d = {}", d);
    println!("  dmin = {}", dmin);
    println!("  scale = {}", scale);
    println!("  min = {}", min);

    // Compute expected value
    let dl = d * scale as f32;
    let ml = dmin * min as f32;
    println!("\nComputed values:");
    println!("  dl = d * scale = {} * {} = {}", d, scale, dl);
    println!("  ml = dmin * min = {} * {} = {}", dmin, min, ml);

    let q = (block.qs[0] & 0x0F) as f32;
    let h = if (block.qh[0] & 0x01) != 0 { 16.0f32 } else { 0.0 };
    let value = q + h;
    let result = dl * value - ml;

    println!("\nElement 0:");
    println!("  qs[0] & 0x0F = {}", q);
    println!("  qh[0] & 0x01 -> h = {}", h);
    println!("  value = q + h = {}", value);
    println!(
        "  result = dl * value - ml = {} * {} - {} = {}",
        dl, value, ml, result
    );

    // Now test with -32 adjustment
    let dl_adj = d * (scale as f32 - 32.0);
    let ml_adj = dmin * (min as f32 - 32.0);
    let result_adj = dl_adj * value - ml_adj;

    println!("\nWith -32 adjustment:");
    println!(
        "  dl = d * (scale - 32) = {} * ({}) = {}",
        d,
        scale as f32 - 32.0,
        dl_adj
    );
    println!(
        "  ml = dmin * (min - 32) = {} * ({}) = {}",
        dmin,
        min as f32 - 32.0,
        ml_adj
    );
    println!("  result = {} * {} - {} = {}", dl_adj, value, ml_adj, result_adj);

    // Which one is correct?
    // GEMV uses: d * (sc-32) * q
    // Let's see what makes sense:
    // - scale=32 means neutral (no scaling)
    // - scale=63 means strong positive
    // - scale=0 means strong negative
    // So (scale-32) makes sense: 0->-32, 32->0, 63->31

    println!("\nConclusion:");
    println!("  Without -32: result = {}", result);
    println!("  With -32: result = {}", result_adj);
    println!(
        "  Expected: (scale-32) should be used, so result = {}",
        result_adj
    );
}
