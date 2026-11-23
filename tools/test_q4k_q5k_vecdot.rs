// Test Q4_K and Q5_K vec_dot implementations
// Create test vectors and compare with expected results

const KMASK1: u32 = 0x3f3f3f3f;
const KMASK2: u32 = 0x0f0f0f0f;
const KMASK3: u32 = 0x03030303;

/// f16 to f32 conversion (simple implementation)
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as i32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x3FF) as i32;

    if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            if sign == 0 { 0.0f32 } else { -0.0f32 }
        } else {
            // Subnormal
            let e = mantissa.leading_zeros() as i32 - 21;
            let m = mantissa << (e + 1);
            let val = (m as f32) * 2.0f32.powi(e - 14);
            if sign == 0 { val } else { -val }
        }
    } else if exp == 31 {
        // Infinity or NaN
        if mantissa == 0 {
            if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY }
        } else {
            f32::NAN
        }
    } else {
        // Normal
        let val = ((mantissa + 1024) as f32) * 2.0f32.powi(exp - 15);
        if sign == 0 { val } else { -val }
    }
}

/// Reference implementation matching llama.cpp
fn vec_dot_q4_k_q8_k_ref(
    x_qs: &[u8; 128],
    y_qs: &[i8; 256],
    y_bsums: &[i16; 16],
    y_d: f32,
    x_d: u16,
    x_dmin: u16,
    x_scales: &[u8; 12]
) -> f32 {
    // Decode scales and mins
    let mut utmp: [u32; 4] = [0; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(x_scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
    }

    // llama.cpp decoding
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let scales: [u8; 8] = unsafe {
        let ptr = utmp.as_ptr() as *const u8;
        [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3),
         *ptr.add(4), *ptr.add(5), *ptr.add(6), *ptr.add(7)]
    };

    let mins: [u8; 8] = unsafe {
        let ptr = utmp.as_ptr() as *const u8;
        [*ptr.add(8), *ptr.add(9), *ptr.add(10), *ptr.add(11),
         *ptr.add(12), *ptr.add(13), *ptr.add(14), *ptr.add(15)]
    };

    // Calculate min correction
    let mut sumi_mins: i32 = 0;
    for j in 0..16 {
        sumi_mins += y_bsums[j] as i32 * mins[j / 2] as i32;
    }

    // Dequantize Q4 values (this is the key part - Q4 uses 4-bit unsigned values)
    let mut aux8 = [0i8; 256];
    for j in 0..4 {
        for l in 0..32 {
            aux8[j * 64 + l] = (x_qs[j * 32 + l] & 0x0F) as i8;
        }
        for l in 0..32 {
            aux8[j * 64 + 32 + l] = (x_qs[j * 32 + l] >> 4) as i8;
        }
    }

    // Dot product with scales (8 groups of 32 values)
    let mut aux32 = [0i32; 8];
    let mut is = 0;
    for j in 0..8 {
        let scale = scales[is] as i32;
        is += 1;
        for l in 0..4 {
            for k in 0..8 {
                let idx = j * 32 + l * 8 + k;
                aux32[l] += scale * y_qs[idx] as i32 * aux8[idx] as i32;
            }
        }
    }

    let d = y_d * f16_to_f32(x_d);
    let dmin = y_d * f16_to_f32(x_dmin);

    let mut sumf = 0.0f32;
    for l in 0..8 {
        sumf += d * aux32[l] as f32;
    }
    sumf -= dmin * sumi_mins as f32;

    sumf
}

/// My NEON implementation simulation (matching current code)
fn vec_dot_q4_k_my_impl(
    x_qs: &[u8; 128],
    y_qs: &[i8; 256],
    y_bsums: &[i16; 16],
    y_d: f32,
    x_d: u16,
    x_dmin: u16,
    x_scales: &[u8; 12]
) -> f32 {
    // Decode scales and mins (my implementation)
    let mut utmp: [u32; 4] = [0; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(x_scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
    }

    // Extract mins first (my buggy way?)
    let mins8_0 = utmp[1] & KMASK1;
    let mins8_1 = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);

    // Now modify utmp for scales
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[0] &= KMASK1;

    // scales from utmp[0] and utmp[1]
    let scales: [u8; 8] = unsafe {
        let ptr = utmp.as_ptr() as *const u8;
        [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3),
         *ptr.add(4), *ptr.add(5), *ptr.add(6), *ptr.add(7)]
    };

    // mins from mins8_0 and mins8_1 (4 bytes each)
    let mins: [u8; 8] = {
        let mut arr = [0u8; 8];
        arr[0] = mins8_0 as u8;
        arr[1] = (mins8_0 >> 8) as u8;
        arr[2] = (mins8_0 >> 16) as u8;
        arr[3] = (mins8_0 >> 24) as u8;
        arr[4] = mins8_1 as u8;
        arr[5] = (mins8_1 >> 8) as u8;
        arr[6] = (mins8_1 >> 16) as u8;
        arr[7] = (mins8_1 >> 24) as u8;
        arr
    };

    // bsums pairwise add (matching vpaddq_s16 behavior)
    let q8sums: [i16; 8] = [
        y_bsums[0] + y_bsums[1],
        y_bsums[2] + y_bsums[3],
        y_bsums[4] + y_bsums[5],
        y_bsums[6] + y_bsums[7],
        y_bsums[8] + y_bsums[9],
        y_bsums[10] + y_bsums[11],
        y_bsums[12] + y_bsums[13],
        y_bsums[14] + y_bsums[15],
    ];

    // Calculate min correction
    let prod_low = (q8sums[0] as i32 * mins[0] as i32) + (q8sums[1] as i32 * mins[1] as i32);
    let prod_high = (q8sums[4] as i32 * mins[4] as i32) + (q8sums[5] as i32 * mins[5] as i32);
    let sumi_mins = prod_low + prod_high;

    // Dequantize and dot product (my way)
    let mut sumi1 = 0i32;
    let mut sumi2 = 0i32;

    for j in 0..4 {
        // Low nibbles
        let mut dot_low = 0i32;
        for l in 0..32 {
            let q4_val = (x_qs[j * 32 + l] & 0x0F) as i8;
            let q8_val = y_qs[j * 64 + l];
            dot_low += q4_val as i32 * q8_val as i32;
        }
        sumi1 += dot_low * scales[j * 2] as i32;

        // High nibbles
        let mut dot_high = 0i32;
        for l in 0..32 {
            let q4_val = (x_qs[j * 32 + l] >> 4) as i8;
            let q8_val = y_qs[j * 64 + 32 + l];
            dot_high += q4_val as i32 * q8_val as i32;
        }
        sumi2 += dot_high * scales[j * 2 + 1] as i32;
    }

    let d = y_d * f16_to_f32(x_d);
    let dmin = y_d * f16_to_f32(x_dmin);

    d * (sumi1 + sumi2) as f32 - dmin * sumi_mins as f32
}

fn main() {
    println!("Testing Q4_K vec_dot implementations\n");

    // Test 1: Simple values
    let mut x_qs = [0u8; 128];
    let y_qs = [1i8; 256];
    let y_bsums = [16i16; 16];
    let y_d = 1.0f32;
    let x_d: u16 = 0x3C00; // f16 1.0
    let x_dmin: u16 = 0x0000; // f16 0.0
    let x_scales = [1u8; 12];

    // With all Q4 values being 0, the result should be 0
    let ref_result = vec_dot_q4_k_q8_k_ref(&x_qs, &y_qs, &y_bsums, y_d, x_d, x_dmin, &x_scales);
    let my_result = vec_dot_q4_k_my_impl(&x_qs, &y_qs, &y_bsums, y_d, x_d, x_dmin, &x_scales);
    println!("Test 1 (all zeros):");
    println!("  Reference: {}", ref_result);
    println!("  My impl:   {}", my_result);

    // Test 2: Q4 values = 1
    for i in 0..128 {
        x_qs[i] = 0x11;  // Both nibbles = 1
    }

    let ref_result = vec_dot_q4_k_q8_k_ref(&x_qs, &y_qs, &y_bsums, y_d, x_d, x_dmin, &x_scales);
    let my_result = vec_dot_q4_k_my_impl(&x_qs, &y_qs, &y_bsums, y_d, x_d, x_dmin, &x_scales);
    println!("\nTest 2 (all ones):");
    println!("  Reference: {}", ref_result);
    println!("  My impl:   {}", my_result);

    // Test 3: Different scales
    let x_scales2: [u8; 12] = [0x15, 0x0A, 0x1E, 0x2B, 0x33, 0x1F, 0x0C, 0x28, 0x19, 0x0D, 0x21, 0x2E];
    for i in 0..128 {
        x_qs[i] = ((i * 17 + 31) % 256) as u8;
    }

    let ref_result = vec_dot_q4_k_q8_k_ref(&x_qs, &y_qs, &y_bsums, y_d, x_d, x_dmin, &x_scales2);
    let my_result = vec_dot_q4_k_my_impl(&x_qs, &y_qs, &y_bsums, y_d, x_d, x_dmin, &x_scales2);
    println!("\nTest 3 (different scales):");
    println!("  Reference: {}", ref_result);
    println!("  My impl:   {}", my_result);

    // Test bsums/mins calculation
    println!("\n=== Testing bsums/mins calculation ===");

    let test_bsums: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let test_mins: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

    // llama.cpp way: bsums[j] * mins[j/2]
    let mut sum_llama = 0i32;
    for j in 0..16 {
        sum_llama += test_bsums[j] as i32 * test_mins[j / 2] as i32;
    }
    println!("llama.cpp bsums/mins: {}", sum_llama);

    // My way: vpaddq_s16 then multiply
    let q8sums: [i16; 8] = [
        test_bsums[0] + test_bsums[1],  // 1+2=3
        test_bsums[2] + test_bsums[3],  // 3+4=7
        test_bsums[4] + test_bsums[5],  // 5+6=11
        test_bsums[6] + test_bsums[7],  // 7+8=15
        test_bsums[8] + test_bsums[9],  // 9+10=19
        test_bsums[10] + test_bsums[11], // 11+12=23
        test_bsums[12] + test_bsums[13], // 13+14=27
        test_bsums[14] + test_bsums[15], // 15+16=31
    ];
    let mut sum_my = 0i32;
    for i in 0..8 {
        sum_my += q8sums[i] as i32 * test_mins[i] as i32;
    }
    println!("My way (vpaddq): {}", sum_my);

    // The two should be equal!
    println!("Match: {}", sum_llama == sum_my);
}
