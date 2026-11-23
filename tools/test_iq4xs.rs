// Test IQ4_XS vec_dot implementation

/// IQ4_NL lookup table: maps 4-bit values to actual quantized values
const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// f16 to f32 conversion
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as i32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x3FF) as i32;

    if exp == 0 {
        if mantissa == 0 {
            if sign == 0 { 0.0f32 } else { -0.0f32 }
        } else {
            let e = mantissa.leading_zeros() as i32 - 21;
            let m = mantissa << (e + 1);
            let val = (m as f32) * 2.0f32.powi(e - 14);
            if sign == 0 { val } else { -val }
        }
    } else if exp == 31 {
        if mantissa == 0 {
            if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let val = ((mantissa + 1024) as f32) * 2.0f32.powi(exp - 15);
        if sign == 0 { val } else { -val }
    }
}

/// Reference implementation matching llama.cpp
fn vec_dot_iq4_xs_q8_k_ref(
    x_qs: &[u8; 128],
    x_scales_l: &[u8; 4],
    x_scales_h: u16,
    x_d: u16,
    y_qs: &[i8; 256],
    y_d: f32,
) -> f32 {
    let d4d8 = f16_to_f32(x_d) * y_d;
    let mut h = x_scales_h;
    let mut qs = x_qs.as_ptr();
    let mut q8 = y_qs.as_ptr();

    let mut sumf = 0.0f32;

    // QK_K/32 = 8 blocks, step by 2 = 4 iterations
    for ib in (0..8).step_by(2) {
        let ls1 = (x_scales_l[ib / 2] & 0xf) as u32 | ((h as u32) << 4) & 0x30;
        let ls2 = (x_scales_l[ib / 2] >> 4) as u32 | ((h as u32) << 2) & 0x30;
        h >>= 4;

        let d1 = d4d8 * (ls1 as i32 - 32) as f32;
        let d2 = d4d8 * (ls2 as i32 - 32) as f32;

        // First 32 values
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..16 {
            unsafe {
                let q4_val_low = *qs.add(j) & 0xf;
                let q4_val_high = *qs.add(j) >> 4;
                sumi1 += *q8.add(j) as i32 * KVALUES_IQ4NL[q4_val_low as usize] as i32;
                sumi2 += *q8.add(j + 16) as i32 * KVALUES_IQ4NL[q4_val_high as usize] as i32;
            }
        }
        sumf += d1 * (sumi1 + sumi2) as f32;

        unsafe {
            qs = qs.add(16);
            q8 = q8.add(32);
        }

        // Second 32 values
        sumi1 = 0;
        sumi2 = 0;
        for j in 0..16 {
            unsafe {
                let q4_val_low = *qs.add(j) & 0xf;
                let q4_val_high = *qs.add(j) >> 4;
                sumi1 += *q8.add(j) as i32 * KVALUES_IQ4NL[q4_val_low as usize] as i32;
                sumi2 += *q8.add(j + 16) as i32 * KVALUES_IQ4NL[q4_val_high as usize] as i32;
            }
        }
        sumf += d2 * (sumi1 + sumi2) as f32;

        unsafe {
            qs = qs.add(16);
            q8 = q8.add(32);
        }
    }

    sumf
}

/// My implementation (matching current NEON code)
fn vec_dot_iq4_xs_my_impl(
    x_qs: &[u8; 128],
    x_scales_l: &[u8; 4],
    x_scales_h: u16,
    x_d: u16,
    y_qs: &[i8; 256],
    y_d: f32,
) -> f32 {
    let mut h = x_scales_h;
    let mut sumi1 = 0i64;
    let mut sumi2 = 0i64;

    // My loop: QK_K/64 = 4 iterations
    for ib in 0..4 {
        let q4_start = ib * 32;
        let q8_start = ib * 64;

        // My scale extraction
        let ls1 = ((x_scales_l[ib] & 0x0F) as i32 | (((h as i32) << 4) & 0x30)) - 32;
        let ls2 = ((x_scales_l[ib] >> 4) as i32 | (((h as i32) << 2) & 0x30)) - 32;
        h >>= 4;

        // Dot products using lookup table
        let mut dot1 = 0i32;
        let mut dot2 = 0i32;

        for j in 0..16 {
            let q4_val_low = x_qs[q4_start + j] & 0xf;
            let q4_val_high = x_qs[q4_start + j] >> 4;
            dot1 += y_qs[q8_start + j] as i32 * KVALUES_IQ4NL[q4_val_low as usize] as i32;
            dot2 += y_qs[q8_start + 16 + j] as i32 * KVALUES_IQ4NL[q4_val_high as usize] as i32;
        }
        sumi1 += (dot1 as i64) * (ls1 as i64);

        // Next 32 values (same pattern with different scale)
        let mut dot3 = 0i32;
        let mut dot4 = 0i32;
        for j in 0..16 {
            let q4_val_low = x_qs[q4_start + 16 + j] & 0xf;
            let q4_val_high = x_qs[q4_start + 16 + j] >> 4;
            dot3 += y_qs[q8_start + 32 + j] as i32 * KVALUES_IQ4NL[q4_val_low as usize] as i32;
            dot4 += y_qs[q8_start + 48 + j] as i32 * KVALUES_IQ4NL[q4_val_high as usize] as i32;
        }
        sumi2 += (dot3 as i64 + dot4 as i64) * (ls2 as i64);
    }

    f16_to_f32(x_d) * y_d * (sumi1 + sumi2) as f32
}

fn main() {
    println!("Testing IQ4_XS vec_dot implementations\n");

    // Test 1: All zeros
    let x_qs = [0u8; 128];
    let x_scales_l = [0x20, 0x20, 0x20, 0x20]; // scales = 0 (ls - 32 = 0)
    let x_scales_h = 0u16;
    let x_d: u16 = 0x3C00; // f16 1.0
    let y_qs = [1i8; 256];
    let y_d = 1.0f32;

    let ref_result = vec_dot_iq4_xs_q8_k_ref(&x_qs, &x_scales_l, x_scales_h, x_d, &y_qs, y_d);
    let my_result = vec_dot_iq4_xs_my_impl(&x_qs, &x_scales_l, x_scales_h, x_d, &y_qs, y_d);
    println!("Test 1 (all zeros):");
    println!("  Reference: {}", ref_result);
    println!("  My impl:   {}", my_result);

    // Test 2: Simple values
    let x_qs2 = [0x11u8; 128]; // All nibbles = 1
    let x_scales_l2 = [0x21, 0x21, 0x21, 0x21]; // ls1=1, ls2=1 (after -32)
    let x_scales_h2 = 0u16;

    // With q4=1 (index 1), kvalues[1] = -104
    // 256 values, each contributes y_qs[i] * -104 * scale
    let ref_result = vec_dot_iq4_xs_q8_k_ref(&x_qs2, &x_scales_l2, x_scales_h2, x_d, &y_qs, y_d);
    let my_result = vec_dot_iq4_xs_my_impl(&x_qs2, &x_scales_l2, x_scales_h2, x_d, &y_qs, y_d);
    println!("\nTest 2 (all ones, scale=1):");
    println!("  Reference: {}", ref_result);
    println!("  My impl:   {}", my_result);

    // Test 3: Check scale extraction
    println!("\n=== Testing scale extraction ===");
    let test_scales_l: [u8; 4] = [0x12, 0x34, 0x56, 0x78];
    let test_scales_h: u16 = 0x1234;

    // llama.cpp way
    let mut h = test_scales_h;
    for ib in (0..8).step_by(2) {
        let ls1 = (test_scales_l[ib / 2] & 0xf) as u32 | ((h as u32) << 4) & 0x30;
        let ls2 = (test_scales_l[ib / 2] >> 4) as u32 | ((h as u32) << 2) & 0x30;
        h >>= 4;
        println!("llama.cpp ib={} ls1={} (after-32={}) ls2={} (after-32={})",
            ib, ls1, ls1 as i32 - 32, ls2, ls2 as i32 - 32);
    }

    // My way
    let mut h = test_scales_h;
    for ib in 0..4 {
        let ls1 = ((test_scales_l[ib] & 0x0F) as i32 | (((h as i32) << 4) & 0x30)) - 32;
        let ls2 = ((test_scales_l[ib] >> 4) as i32 | (((h as i32) << 2) & 0x30)) - 32;
        h >>= 4;
        println!("My impl ib={} ls1={} ls2={}", ib, ls1, ls2);
    }

    // Test 4: Random values
    println!("\n=== Test 4: Random values ===");
    let mut x_qs3 = [0u8; 128];
    let mut y_qs3 = [0i8; 256];
    for i in 0..128 {
        x_qs3[i] = ((i * 17 + 31) % 256) as u8;
    }
    for i in 0..256 {
        y_qs3[i] = ((i * 13 + 7) % 256) as i8;
    }
    let x_scales_l3: [u8; 4] = [0x15, 0x27, 0x39, 0x4B];
    let x_scales_h3: u16 = 0xABCD;

    let ref_result = vec_dot_iq4_xs_q8_k_ref(&x_qs3, &x_scales_l3, x_scales_h3, x_d, &y_qs3, y_d);
    let my_result = vec_dot_iq4_xs_my_impl(&x_qs3, &x_scales_l3, x_scales_h3, x_d, &y_qs3, y_d);
    println!("Reference: {}", ref_result);
    println!("My impl:   {}", my_result);
}
