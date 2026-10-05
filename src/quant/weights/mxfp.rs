// MXFP formats: MXFP4 (E2M1 with E8M0 scale)

use crate::quant::types::*;

const QK_MXFP: usize = 32;

/// Quantize FP32 to MXFP4 (E2M1 with E8M0 scale)
pub fn quantize_mxfp4(x: &[f32]) -> Vec<BlockMXFP4> {
    assert!(x.len() % QK_MXFP == 0);
    let nb = x.len() / QK_MXFP;
    let mut blocks = Vec::with_capacity(nb);

    for ib in 0..nb {
        let xb = &x[ib * QK_MXFP..(ib + 1) * QK_MXFP];

        // Find max to determine E8M0 scale
        let mut amax = 0.0f32;
        for j in 0..QK_MXFP {
            amax = amax.max(xb[j].abs());
        }

        // E8M0 scale: exponent only, shared across block
        // Scale = 2^(exp-127) where exp is the E8M0 exponent
        let e8m0_exp = if amax > 0.0 {
            // Find optimal exponent
            let log2_max = amax.log2();
            let exp = (log2_max + 127.0 + 2.0).floor() as u8; // +2 for E2M1 range
            exp.min(254).max(1)
        } else {
            127 // scale = 1.0
        };

        // Quantize each value to E2M1 (4-bit)
        // E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let x0 = xb[j];
            let x1 = xb[j + 16];

            let q0 = quantize_e2m1(x0, e8m0_exp);
            let q1 = quantize_e2m1(x1, e8m0_exp);

            qs[j] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        blocks.push(BlockMXFP4 { e: e8m0_exp, qs });
    }
    blocks
}

/// Quantize single value to E2M1
fn quantize_e2m1(x: f32, e8m0_exp: u8) -> u8 {
    if x == 0.0 {
        return 0;
    }

    let sign = if x < 0.0 { 0x08 } else { 0 };
    let ax = x.abs();

    // Compute scale from E8M0
    let scale = 2.0f32.powi((e8m0_exp as i32) - 127 - 2);

    // Normalize to E2M1 range
    let normalized = ax / scale;

    // E2M1 can represent: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    // Exponent bits: 00=0, 01=1, 10=2, 11=4
    // Mantissa bit adds 0.5x
    let exp_bits;
    let mant_bit;

    if normalized >= 4.0 {
        exp_bits = 0x06; // exponent = 3 (value 4)
        mant_bit = if normalized >= 6.0 { 0x01 } else { 0x00 };
    } else if normalized >= 2.0 {
        exp_bits = 0x04; // exponent = 2 (value 2)
        mant_bit = if normalized >= 3.0 { 0x01 } else { 0x00 };
    } else if normalized >= 1.0 {
        exp_bits = 0x02; // exponent = 1 (value 1)
        mant_bit = if normalized >= 1.5 { 0x01 } else { 0x00 };
    } else {
        exp_bits = 0x00; // exponent = 0 (value 0.5)
        mant_bit = if normalized >= 0.25 { 0x01 } else { 0x00 };
    }

    sign | (exp_bits << 1) | mant_bit
}
