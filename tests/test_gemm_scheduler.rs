// Test GEMM scheduler 3-layer fallback

#[cfg(target_arch = "aarch64")]
#[test]
fn test_gemm_path_selection() {
    use infer_train::model::tensor::DataType;
    use infer_train::quant::gemm::{
        GemmPath, requires_vec_dot, select_gemm_path, should_use_blas,
        supports_sgemm,
    };

    // Test supports_sgemm
    assert!(supports_sgemm(DataType::F32));
    assert!(supports_sgemm(DataType::F16));
    assert!(supports_sgemm(DataType::Q4_0));
    assert!(supports_sgemm(DataType::Q5_0));
    assert!(supports_sgemm(DataType::Q8_0));
    assert!(supports_sgemm(DataType::IQ4_NL));

    assert!(!supports_sgemm(DataType::Q4_K));
    assert!(!supports_sgemm(DataType::Q5_K));
    assert!(!supports_sgemm(DataType::Q6_K));

    // Test requires_vec_dot
    assert!(requires_vec_dot(DataType::Q4_K));
    assert!(requires_vec_dot(DataType::Q5_K));
    assert!(requires_vec_dot(DataType::Q6_K));
    assert!(requires_vec_dot(DataType::IQ4_XS));

    assert!(!requires_vec_dot(DataType::F32));
    assert!(!requires_vec_dot(DataType::Q4_0));

    // Test should_use_blas
    // Small batch: SIMD
    assert!(!should_use_blas(1024, 1, 1024));
    assert!(!should_use_blas(1024, 16, 1024));

    // Large batch: BLAS
    assert!(should_use_blas(1024, 32, 1024));
    assert!(should_use_blas(1024, 128, 1024));

    // Test select_gemm_path
    // Non-K-quant, small batch: SIMD
    assert_eq!(select_gemm_path(DataType::F32, 1024, 1, 1024), GemmPath::Simd);
    assert_eq!(
        select_gemm_path(DataType::Q4_0, 1024, 16, 1024),
        GemmPath::Simd
    );

    // Non-K-quant, large batch: BLAS
    assert_eq!(select_gemm_path(DataType::F32, 1024, 32, 1024), GemmPath::Blas);
    assert_eq!(
        select_gemm_path(DataType::Q4_0, 1024, 64, 1024),
        GemmPath::Blas
    );

    // K-quant: always vec_dot
    assert_eq!(
        select_gemm_path(DataType::Q4_K, 1024, 1, 1024),
        GemmPath::VecDot
    );
    assert_eq!(
        select_gemm_path(DataType::Q5_K, 1024, 32, 1024),
        GemmPath::VecDot
    );
    assert_eq!(
        select_gemm_path(DataType::Q6_K, 1024, 128, 1024),
        GemmPath::VecDot
    );
}
