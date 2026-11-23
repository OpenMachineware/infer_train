// Detailed test for IQ4_XS to find the bug

const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

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

/// llama.cpp exact implementation
fn vec_dot_iq4_xs_llama_cpp(
    x_qs: &[u8; 128],
    x_scales_l: &[u8; 4],
    x_scales_h: u16,
    x_d: u16,
    y_qs: &[i8; 256],
    y_d: f32,
) -> f32 {
    let d4d8 = f16_to_f32(x_d) * y_d;
    let mut h = x_scales_h;
    let mut qs = 0usize;  // Index into x_qs
    let mut q8 = 0usize;  // Index into y_qs

    let mut sumf = 0.0f32;

    // ib = 0, 2, 4, 6 (step by 2)
    for ib in (0..8).step_by(2) {
        let ls1 = (x_scales_l[ib / 2] & 0xf) | (((h << 4) & 0x30) as u8);
        let ls2 = (x_scales_l[ib / 2] >> 4) | (((h << 2) & 0x30) as u8);
        h >>= 4;

        let d1 = d4d8 * (ls1 as i32 - 32) as f32;
        let d2 = d4d8 * (ls2 as i32 - 32) as f32;

        println!("ib={} ls1={} ls2={} d1={} d2={}", ib, ls1, ls2, d1, d2);

        // First 32 Q8 values, first 16 QS bytes
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..16 {
            let q4_low = x_qs[qs + j] & 0xf;
            let q4_high = x_qs[qs + j] >> 4;
            sumi1 += y_qs[q8 + j] as i32 * KVALUES_IQ4NL[q4_low as usize] as i32;
            sumi2 += y_qs[q8 + 16 + j] as i32 * KVALUES_IQ4NL[q4_high as usize] as i32;
        }
        sumf += d1 * (sumi1 + sumi2) as f32;
        println!("  First half: qs[{}..{}] q8[{}..{}] sumi1={} sumi2={} sumf={}",
            qs, qs + 16, q8, q8 + 32, sumi1, sumi2, sumf);

        qs += 16;
        q8 += 32;

        // Second 32 Q8 values, next 16 QS bytes
        sumi1 = 0;
        sumi2 = 0;
        for j in 0..16 {
            let q4_low = x_qs[qs + j] & 0xf;
            let q4_high = x_qs[qs + j] >> 4;
            sumi1 += y_qs[q8 + j] as i32 * KVALUES_IQ4NL[q4_low as usize] as i32;
            sumi2 += y_qs[q8 + 16 + j] as i32 * KVALUES_IQ4NL[q4_high as usize] as i32;
        }
        sumf += d2 * (sumi1 + sumi2) as f32;
        println!("  Second half: qs[{}..{}] q8[{}..{}] sumi1={} sumi2={} sumf={}",
            qs, qs + 16, q8, q8 + 32, sumi1, sumi2, sumf);

        qs += 16;
        q8 += 32;
    }

    sumf
}

/// My implementation
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

    for ib in 0..4 {
        let q4_start = ib * 32;
        let q8_start = ib * 64;

        let ls1 = ((x_scales_l[ib] & 0x0F) as i32 | (((h as i32) << 4) & 0x30)) - 32;
        let ls2 = ((x_scales_l[ib] >> 4) as i32 | (((h as i32) << 2) & 0x30)) - 32;
        h >>= 4;

        println!("ib={} ls1={} ls2={}", ib, ls1, ls2);

        // My way: first 32 Q8 values with ls1
        let mut dot1 = 0i32;
        let mut dot2 = 0i32;
        for j in 0..16 {
            let q4_low = x_qs[q4_start + j] & 0xf;
            let q4_high = x_qs[q4_start + j] >> 4;
            dot1 += y_qs[q8_start + j] as i32 * KVALUES_IQ4NL[q4_low as usize] as i32;
            dot2 += y_qs[q8_start + 16 + j] as i32 * KVALUES_IQ4NL[q4_high as usize] as i32;
        }
        sumi1 += (dot1 + dot2) as i64 * ls1 as i64;
        println!("  First half: qs[{}..{}] q8[{}..{}] dot1={} dot2={} sumi1={}",
            q4_start, q4_start + 16, q8_start, q8_start + 32, dot1, dot2, sumi1);

        // Next 32 Q8 values with ls2
        let mut dot3 = 0i32;
        let mut dot4 = 0i32;
        for j in 0..16 {
            let q4_low = x_qs[q4_start + 16 + j] & 0xf;
            let q4_high = x_qs[q4_start + 16 + j] >> 4;
            dot3 += y_qs[q8_start + 32 + j] as i32 * KVALUES_IQ4NL[q4_low as usize] as i32;
            dot4 += y_qs[q8_start + 48 + j] as i32 * KVALUES_IQ4NL[q4_high as usize] as i32;
        }
        sumi2 += (dot3 + dot4) as i64 * ls2 as i64;
        println!("  Second half: qs[{}..{}] q8[{}..{}] dot3={} dot4={} sumi2={}",
            q4_start + 16, q4_start + 32, q8_start + 32, q8_start + 64, dot3, dot4, sumi2);
    }

    f16_to_f32(x_d) * y_d * (sumi1 + sumi2) as f32
}

fn main() {
    println!("Detailed comparison of IQ4_XS implementations\n");

    // Use simple test data
    let x_qs: [u8; 128] = [0x11; 128]; // All nibbles = 1
    let x_scales_l: [u8; 4] = [0x21, 0x21, 0x21, 0x21]; // ls1=1, ls2=1 after -32
    let x_scales_h: u16 = 0u16;
    let x_d: u16 = 0x3C00; // f16 1.0
    let y_qs: [i8; 256] = [1i8; 256];
    let y_d = 1.0f32;

    println!("=== llama.cpp implementation ===\n");
    let llama_result = vec_dot_iq4_xs_llama_cpp(&x_qs, &x_scales_l, x_scales_h, x_d, &y_qs, y_d);
    println!("\nFinal result: {}", llama_result);

    println!("\n=== My implementation ===\n");
    let my_result = vec_dot_iq4_xs_my_impl(&x_qs, &x_scales_l, x_scales_h, x_d, &y_qs, y_d);
    println!("\nFinal result: {}", my_result);

    println!("\n=== Difference ===");
    println!("llama.cpp: {}", llama_result);
    println!("My impl:   {}", my_result);
    println!("Diff: {}", llama_result - my_result);
}
