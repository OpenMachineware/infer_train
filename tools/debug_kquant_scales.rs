// Debug K-quant scales/mins encoding
// Compare my implementation with llama.cpp reference

use std::mem;

const KMASK1: u32 = 0x3f3f3f3f;
const KMASK2: u32 = 0x0f0f0f0f;
const KMASK3: u32 = 0x03030303;

/// llama.cpp reference implementation for Q4_K/Q5_K scales decoding
fn decode_scales_llama_cpp(scales_bytes: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut utmp: [u32; 4] = [0; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            scales_bytes.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );
    }

    // llama.cpp decoding:
    // utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    // uaux = utmp[1] & kmask1;
    // utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    // utmp[2] = uaux;
    // utmp[0] &= kmask1;

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    // Now: utmp[0-3] each contain 4 bytes
    // scales = bytes 0-7 (utmp[0] and utmp[1])
    // mins = bytes 8-15 (utmp[2] and utmp[3])
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

    (scales, mins)
}

/// My current Q4_K implementation (potentially buggy)
fn decode_scales_my_q4k(scales_bytes: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut utmp: [u32; 4] = [0; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            scales_bytes.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );
    }

    // My buggy implementation (missing utmp[3] calculation!)
    // Extract mins first (before modifying utmp)
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

    (scales, mins)
}

/// My current Q5_K implementation
fn decode_scales_my_q5k(scales_bytes: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut utmp: [u32; 4] = [0; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            scales_bytes.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );
    }

    // My Q5_K implementation (matches llama.cpp)
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    // scales = bytes 0-7
    // mins = bytes 8-15
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

    (scales, mins)
}

fn main() {
    // Create test data: 12 bytes with known values
    // Each 6-bit value is encoded, 8 scales + 8 mins = 16 values in 12 bytes

    // Test case 1: simple pattern
    let test1: [u8; 12] = [
        0x3F, 0x3F, 0x3F, 0x3F,  // scales 0-3: all 63
        0x3F, 0x3F, 0x3F, 0x3F,  // scales 4-7: all 63
        0x3F, 0x3F, 0x3F, 0x3F,  // mins 0-3: all 63
    ];

    // Test case 2: mixed values
    let test2: [u8; 12] = [
        0x01, 0x23, 0x45, 0x67,  // mixed
        0x89, 0xAB, 0xCD, 0xEF,  // mixed
        0x11, 0x22, 0x33, 0x44,  // mixed
    ];

    // Test case 3: from actual Qwen3 model (would need real data)
    let test3: [u8; 12] = [
        0x15, 0x0A, 0x1E, 0x2B,  // typical scale values
        0x33, 0x1F, 0x0C, 0x28,
        0x19, 0x0D, 0x21, 0x2E,
    ];

    for (idx, test) in [test1, test2, test3].iter().enumerate() {
        println!("\n=== Test case {} ===", idx + 1);
        println!("Input bytes: {:02X?}", test);

        let (scales_llama, mins_llama) = decode_scales_llama_cpp(test);
        println!("\nllama.cpp reference:");
        println!("  scales: {:?}", scales_llama);
        println!("  mins:   {:?}", mins_llama);

        let (scales_my_q4k, mins_my_q4k) = decode_scales_my_q4k(test);
        println!("\nMy Q4_K implementation:");
        println!("  scales: {:?}", scales_my_q4k);
        println!("  mins:   {:?}", mins_my_q4k);
        let scales_match = scales_llama == scales_my_q4k;
        let mins_match = mins_llama == mins_my_q4k;
        println!("  scales match: {}, mins match: {}", scales_match, mins_match);

        let (scales_my_q5k, mins_my_q5k) = decode_scales_my_q5k(test);
        println!("\nMy Q5_K implementation:");
        println!("  scales: {:?}", scales_my_q5k);
        println!("  mins:   {:?}", mins_my_q5k);
        let scales_match = scales_llama == scales_my_q5k;
        let mins_match = mins_llama == mins_my_q5k;
        println!("  scales match: {}, mins match: {}", scales_match, mins_match);
    }

    // Test with random bytes
    println!("\n=== Random test (100 samples) ===");
    let mut mismatches_q4k = 0;
    let mut mismatches_q5k = 0;

    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    for i in 0..100 {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(i);
        let hash = hasher.finish();

        let mut test: [u8; 12] = [0; 12];
        for j in 0..12 {
            test[j] = ((hash >> (j * 5)) & 0xFF) as u8;
        }

        let (s_llama, m_llama) = decode_scales_llama_cpp(&test);
        let (s_q4k, m_q4k) = decode_scales_my_q4k(&test);
        let (s_q5k, m_q5k) = decode_scales_my_q5k(&test);

        if s_llama != s_q4k || m_llama != m_q4k {
            mismatches_q4k += 1;
        }
        if s_llama != s_q5k || m_llama != m_q5k {
            mismatches_q5k += 1;
        }
    }

    println!("Q4_K mismatches: {}/100", mismatches_q4k);
    println!("Q5_K mismatches: {}/100", mismatches_q5k);
}
