// Debug Q4_1 attention error
use infer_train::quant::kv_cache::quant::{
    quantize_row_q4_1, quantize_row_q8_1,
};
use infer_train::quant::kv_cache::{
    BlockQ4_1 as BlockQ4_1KV, BlockQ8_1 as BlockQ8_1KV, QK,
};
use infer_train::quant::types::{BlockQ4_1, BlockQ8_1};
use infer_train::quant::vec_dot::arm::q_0_1::vec_dot_q4_1_q8_1_neon;

fn main() {
    println!("=== Q4_1 Debug ===\n");

    let n = 32;

    // Simple test: create known values
    let k_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let q_vals: Vec<f32> =
        (0..n).map(|i| ((i as f32) * 0.1 + 0.5).min(1.0)).collect();

    // Quantize K to Q4_1 (using kv_cache types)
    let mut k_q4_1_kv = vec![BlockQ4_1KV { d: 0, m: 0, qs: [0u8; QK / 2] }; 1];
    quantize_row_q4_1(&k_vals, &mut k_q4_1_kv);

    // Quantize Q to Q8_1 (using kv_cache types)
    let mut q_q8_1_kv = vec![BlockQ8_1KV { d: 0, s: 0, qs: [0i8; QK] }; 1];
    quantize_row_q8_1(&q_vals, &mut q_q8_1_kv);

    // Transmute to types module blocks (same repr(C))
    let k_q4_1: &[BlockQ4_1] = unsafe {
        std::slice::from_raw_parts(
            k_q4_1_kv.as_ptr() as *const BlockQ4_1,
            k_q4_1_kv.len(),
        )
    };
    let q_q8_1: &[BlockQ8_1] = unsafe {
        std::slice::from_raw_parts(
            q_q8_1_kv.as_ptr() as *const BlockQ8_1,
            q_q8_1_kv.len(),
        )
    };

    println!("K values: {:?}", &k_vals[..10.min(n)]);
    println!("Q values: {:?}", &q_vals[..10.min(n)]);

    // Dequantize K to check
    let k_deq: Vec<f32> = (0..n)
        .map(|i| {
            let packed = k_q4_1[0].qs[i / 2];
            let nibble = if i % 2 == 0 { packed & 0x0F } else { packed >> 4 };
            let scale = half::f16::from_bits(k_q4_1[0].d).to_f32();
            let min = half::f16::from_bits(k_q4_1[0].m).to_f32();
            scale * (nibble as f32) + min
        })
        .collect();

    println!("K dequantized: {:?}", &k_deq[..10.min(n)]);

    // Dequantize Q to check
    let q_deq: Vec<f32> = (0..n)
        .map(|i| {
            let q = q_q8_1[0].qs[i];
            let scale = half::f16::from_bits(q_q8_1[0].d).to_f32();
            (q as f32) * scale
        })
        .collect();

    println!("Q dequantized: {:?}", &q_deq[..10.min(n)]);

    // Compute reference dot product
    let ref_dot: f32 =
        k_vals.iter().zip(q_vals.iter()).map(|(k, q)| k * q).sum();
    println!("\nReference dot product: {}", ref_dot);

    // Compute dequantized dot product
    let deq_dot: f32 = k_deq.iter().zip(q_deq.iter()).map(|(k, q)| k * q).sum();
    println!("Dequantized dot product: {}", deq_dot);

    // Compute using NEON vec_dot
    let neon_dot = unsafe { vec_dot_q4_1_q8_1_neon(n, k_q4_1, q_q8_1) };
    println!("NEON vec_dot: {}", neon_dot);

    // Check formula: sum = d_k * d_q * sum(q_i * k_i) + s_q * m_k
    let d_k = half::f16::from_bits(k_q4_1[0].d).to_f32();
    let m_k = half::f16::from_bits(k_q4_1[0].m).to_f32();
    let d_q = half::f16::from_bits(q_q8_1[0].d).to_f32();
    let s_q = half::f16::from_bits(q_q8_1[0].s).to_f32();

    println!("\nK scale (d_k): {}", d_k);
    println!("K min (m_k): {}", m_k);
    println!("Q scale (d_q): {}", d_q);
    println!("Q sum (s_q): {}", s_q);

    // Manual compute
    let mut dot_raw = 0i32;
    for j in 0..16 {
        let k0 = (k_q4_1[0].qs[j] & 0x0F) as i32;
        let k1 = (k_q4_1[0].qs[j] >> 4) as i32;
        let q0 = q_q8_1[0].qs[j] as i32;
        let q1 = q_q8_1[0].qs[j + 16] as i32;
        dot_raw += k0 * q0 + k1 * q1;
    }
    println!("\nRaw dot product (int32): {}", dot_raw);

    let expected = d_k * d_q * (dot_raw as f32) + s_q * m_k;
    println!("Expected result: {}", expected);

    // Check error
    println!("\nError (NEON vs reference): {}", (neon_dot - ref_dot).abs());
    println!("Error (dequantized vs reference): {}", (deq_dot - ref_dot).abs());
}
