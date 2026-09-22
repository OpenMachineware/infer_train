mod tests {
    use infer_train::quant::vec_dot::scalar::fp32::vec_dot_fp32;
    use infer_train::quant::vec_dot::scalar::fp16::vec_dot_fp16;
    use infer_train::quant::vec_dot::scalar::bf16::vec_dot_bf16;

    fn generate_test_data(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.1 + 2.0 * ((i as f32).cos())).collect()
    }

    #[test]
    fn test_fp32_vec_dot() {
        let a = generate_test_data(256);
        let b = generate_test_data(128);

        let result = vec_dot_fp32(&a, &b);

        // Reference calculation
        let ref_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        println!("FP32 scalar: {} (ref: {})", result, ref_result);
        assert!((result - ref_result).abs() < 1e-6);
    }

    #[test]
    fn test_fp16_vec_dot() {
        let a: Vec<half::f16> = generate_test_data(256).iter().map(|&x| half::f16::from_f32(x)).collect();
        let b: Vec<half::f16> = generate_test_data(128).iter().map(|&x| half::f16::from_f32(x)).collect();

        let result = vec_dot_fp16(&a, &b);

        // Reference calculation
        let ref_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x.to_f32() * y.to_f32()).sum();

        println!("FP16 scalar: {} (ref: {})", result, ref_result);
        assert!((result - ref_result).abs() < 1e-3); // FP16 has lower precision
    }

    #[test]
    fn test_bf16_vec_dot() {
        let a: Vec<half::bf16> = generate_test_data(256).iter().map(|&x| half::bf16::from_f32(x)).collect();
        let b: Vec<half::bf16> = generate_test_data(128).iter().map(|&x| half::bf16::from_f32(x)).collect();

        let result = vec_dot_bf16(&a, &b);

        // Reference calculation
        let ref_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x.to_f32() * y.to_f32()).sum();

        println!("BF16 scalar: {} (ref: {})", result, ref_result);
        assert!((result - ref_result).abs() < 1e-2); // BF16 has lower precision
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_fp32_neon() {
        use infer_train::quant::vec_dot::arm::vec_dot_fp32_neon;

        let a = generate_test_data(256);
        let b = generate_test_data(256);

        let result = unsafe { vec_dot_fp32_neon(&a, &b) };
        let ref_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        println!("FP32 NEON: {} (ref: {})", result, ref_result);
        assert!((result - ref_result).abs() < 1e-4);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_fp16_neon() {
        use infer_train::quant::vec_dot::arm::vec_dot_fp16_neon;

        let a: Vec<u16> = generate_test_data(256).iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let b: Vec<u16> = generate_test_data(256).iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();

        let result = unsafe { vec_dot_fp16_neon(&a, &b) };
        let ref_result: f32 = a.iter().zip(b.iter())
            .map(|(&x, &y)| half::f16::from_bits(x).to_f32() * half::f16::from_bits(y).to_f32())
            .sum();

        println!("FP16 NEON: {} (ref: {})", result, ref_result);
        assert!((result - ref_result).abs() < 1e-3);
    }
}
