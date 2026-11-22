#[test]
fn test_q8k_quantization() {
    use infer_train::backend::arm::quant::quantize_q8_k;
    use infer_train::types::{BlockQ8_K, QK_K};

    // Test with known values
    let test_values: Vec<f32> =
        (0..256).map(|i| (i as f32 - 128.0) / 100.0).collect();

    let mut q8_blocks =
        vec![BlockQ8_K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; 1];
    quantize_q8_k(&test_values, &mut q8_blocks);

    println!("d = {:.6}", q8_blocks[0].d);
    println!("qs[:10] = {:?}", &q8_blocks[0].qs[..10]);
    println!("bsums[:4] = {:?}", &q8_blocks[0].bsums[..4]);

    // Verify bsums
    for j in 0..4 {
        let expected_sum: i32 = q8_blocks[0].qs[j * 16..(j + 1) * 16]
            .iter()
            .map(|&q| q as i32)
            .sum();
        assert_eq!(
            q8_blocks[0].bsums[j], expected_sum as i16,
            "bsums[{}] mismatch",
            j
        );
    }
}
