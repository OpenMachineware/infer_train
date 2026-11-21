//! End-to-end test for Q6_K vec_dot

use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use infer_train::types::{BlockQ6_K, BlockQ8_K};

fn main() {
    // Test 1: All zeros -> q6=0, after -32=-32
    let mut q6 = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16],
        d: 15360u16, // FP16 for 1.0
    };

    let mut q8 = BlockQ8_K { d: 1.0f32, qs: [1i8; 256], bsums: [16i16; 16] };

    let result = vec_dot_q6_k_q8_k(&[q6], &[q8]);
    println!("Test 1: ql=0, qh=0, q8=1, scale=1");
    println!("  Expected: -8192 (q6=0, after -32=-32, * 256)");
    println!("  Got: {}", result);
    println!();

    // Test 2: qh=170 (all high bits = 2) -> q6=32, after -32=0
    for i in 0..64 {
        q6.qh[i] = 170;
    }

    let result = vec_dot_q6_k_q8_k(&[q6], &[q8]);
    println!("Test 2: qh=170 (all high bits=2)");
    println!("  Expected: 0 (q6=32, after -32=0)");
    println!("  Got: {}", result);
    println!();

    // Test 3: ql[0]=1, qh[0]=2 -> a[0] = 33, after -32=1
    q6.ql[0] = 1;
    q6.qh[0] = 2;
    for i in 1..64 {
        q6.qh[i] = 170; // Reset to 32 for rest
    }

    let result = vec_dot_q6_k_q8_k(&[q6], &[q8]);
    println!("Test 3: ql[0]=1, qh[0]=2 (first element = 33)");
    println!("  Expected: 1 (first element contributes 1, rest contribute 0)");
    println!("  Got: {}", result);
    println!();

    // Test 4: Different scales
    q6.scales[0] = 2; // Scale for first 16 elements
    let result = vec_dot_q6_k_q8_k(&[q6], &[q8]);
    println!("Test 4: scale[0]=2");
    println!("  First 16 elements have q6=33->1, scaled by 2");
    println!(
        "  Expected: 16 (first 16 contribute 1*2*16=32, next 16 contribute 1*16=16, rest 0...)"
    );
    println!("  Wait, let me recalculate...");
    println!("  First 16: q6=33->1, scale=2, q8=1 -> 1*2*16 = 32");
    println!("  Next 16: q6=32->0, scale=1, q8=1 -> 0*1*16 = 0");
    println!("  But we also have scales for other elements...");
    println!("  Got: {}", result);

    // Check FP16 conversion
    println!("\nFP16 value: {}", q6.d);
}
