// Test IQ4_XS with different scale values to find discrepancy

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

/// llama.cpp implementation (exact copy)
fn vec_dot_iq4_xs_llama(
    x_qs: &[u8; 128],
    x_scales_l: &[u8; 4],
    x_scales_h: u16,
    x_d: u16,
    y_qs: &[i8; 256],
    y_d: f32,
) -> f32 {
    let d4d8 = f16_to_f32(x_d) * y_d;
    let mut h = x_scales_h;
    let mut qs = 0usize;
    let mut q8 = 0usize;

    let mut sumf = 0.0f32;

    for ib in (0..8).step_by(2) {
        let ls1 = (x_scales_l[ib / 2] & 0xf) | (((h << 4) & 0x30) as u8);
        let ls2 = (x_scales_l[ib / 2] >> 4) | (((h << 2) & 0x30) as u8);
        h >>= 4;

        let d1 = d4d8 * (ls1 as i32 - 32) as f32;
        let d2 = d4d8 * (ls2 as i32 - 32) as f32;

        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..16 {
            sumi1 += y_qs[q8 + j] as i32 * KVALUES_IQ4NL[x_qs[qs + j] as usize & 0xf] as i32;
            sumi2 += y_qs[q8 + 16 + j] as i32 * KVALUES_IQ4NL[x_qs[qs + j] as usize >> 4] as i32;
        }
        sumf += d1 * (sumi1 + sumi2) as f32;

        qs += 16;
        q8 += 32;

        sumi1 = 0;
        sumi2 = 0;
        for j in 0..16 {
            sumi1 += y_qs[q8 + j] as i32 * KVALUES_IQ4NL[x_qs[qs + j] as usize & 0xf] as i32;
            sumi2 += y_qs[q8 + 16 + j] as i32 * KVALUES_IQ4NL[x_qs[qs + j] as usize >> 4] as i32;
        }
        sumf += d2 * (sumi1 + sumi2) as f32;

        qs += 16;
        q8 += 32;
    }

    sumf
}

/// My NEON implementation (matching current code)
fn vec_dot_iq4_xs_my(
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

        let mut dot1 = 0i32;
        let mut dot2 = 0i32;
        for j in 0..16 {
            dot1 += y_qs[q8_start + j] as i32 * KVALUES_IQ4NL[x_qs[q4_start + j] as usize & 0xf] as i32;
            dot2 += y_qs[q8_start + 16 + j] as i32 * KVALUES_IQ4NL[x_qs[q4_start + j] as usize >> 4] as i32;
        }
        sumi1 += (dot1 + dot2) as i64 * ls1 as i64;

        let mut dot3 = 0i32;
        let mut dot4 = 0i32;
        for j in 0..16 {
            dot3 += y_qs[q8_start + 32 + j] as i32 * KVALUES_IQ4NL[x_qs[q4_start + 16 + j] as usize & 0xf] as i32;
            dot4 += y_qs[q8_start + 48 + j] as i32 * KVALUES_IQ4NL[x_qs[q4_start + 16 + j] as usize >> 4] as i32;
        }
        sumi2 += (dot3 + dot4) as i64 * ls2 as i64;
    }

    f16_to_f32(x_d) * y_d * (sumi1 + sumi2) as f32
}

fn main() {
    println!("Testing various scale combinations\n");

    let x_d: u16 = 0x3C00; // f16 1.0
    let y_d = 1.0f32;
    let x_qs: [u8; 128] = [0x11; 128];
    let y_qs: [i8; 256] = [1i8; 256];

    // Test different scales_l values
    let test_cases: Vec<([u8; 4], u16, &str)> = vec![
        ([0x21, 0x21, 0x21, 0x21], 0, "ls1=1, ls2=1 after -32"),
        ([0x20, 0x20, 0x20, 0x20], 0, "scales_l=0x20, scales_h=0"),
        ([0x00, 0x00, 0x00, 0x00], 0, "all zeros"),
        ([0xFF, 0xFF, 0xFF, 0xFF], 0, "all 0xFF"),
        ([0x15, 0x27, 0x39, 0x4B], 0xABCD, "random values"),
    ];

    for (scales_l, scales_h, desc) in &test_cases {
        let llama_result = vec_dot_iq4_xs_llama(&x_qs, scales_l, *scales_h, x_d, &y_qs, y_d);
        let my_result = vec_dot_iq4_xs_my(&x_qs, scales_l, *scales_h, x_d, &y_qs, y_d);
        let diff = (llama_result - my_result).abs();

        println!("Test: {}", desc);
        println!("  scales_l: {:?}", scales_l);
        println!("  scales_h: 0x{:04X}", scales_h);
        println!("  llama.cpp: {}", llama_result);
        println!("  my impl:   {}", my_result);
        println!("  diff: {} {}", diff, if diff.abs() < 0.01 { "OK" } else { "MISMATCH!" });
        println!();
    }

    // Now test with all Q values being 0
    println!("=== Testing with x_qs = 0 ===\n");
    let x_qs_zero: [u8; 128] = [0u8; 128];

    for (scales_l, scales_h, desc) in &test_cases {
        let llama_result = vec_dot_iq4_xs_llama(&x_qs_zero, scales_l, *scales_h, x_d, &y_qs, y_d);
        let my_result = vec_dot_iq4_xs_my(&x_qs_zero, scales_l, *scales_h, x_d, &y_qs, y_d);
        let diff = (llama_result - my_result).abs();

        println!("Test: {}", desc);
        println!("  llama.cpp: {}", llama_result);
        println!("  my impl:   {}", my_result);
        println!("  diff: {} {}", diff, if diff.abs() < 0.01 { "OK" } else { "MISMATCH!" });
        println!();
    }
}
