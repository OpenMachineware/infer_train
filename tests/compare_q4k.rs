// Compare standard GEMV vs interleaved GEMV for Q4_K
use infer_train::backend::arm::{gemv, linear, repack, vec_dot};
use infer_train::model::GgufFile;
use infer_train::types::{BlockQ4_K, BlockQ4_Kx8, BlockQ8_K, QK_K};

#[test]
fn compare_q4k_gemv_test() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load model");

    // Get blk.24.attn_q.weight (Q4_K)
    let tensor_name = "blk.24.attn_q.weight";
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == tensor_name)
        .unwrap_or_else(|| {
            // Fallback to any Q4_K tensor
            gguf.tensors
                .iter()
                .find(|t| t.quant_type == infer_train::types::QuantType::Q4_K)
                .expect("No Q4_K tensor found")
        });

    let weight_data =
        gguf.get_tensor_data(&tensor.name).expect("Failed to get tensor data");

    let in_features = tensor.dims.get(0).copied().unwrap_or(0) as usize;
    let out_features = tensor.dims.get(1).copied().unwrap_or(0) as usize;
    let stride = tensor.strides.get(1).copied().unwrap_or(0) as usize;
    let n_blocks = in_features / QK_K;

    println!("Tensor: {}", tensor.name);
    println!("  dims: {:?}", tensor.dims);
    println!("  quant_type: {:?}", tensor.quant_type);
    println!("  in_features: {}, out_features: {}", in_features, out_features);

    // Create test input (Q8_K)
    let input_f32: Vec<f32> =
        (0..in_features).map(|i| ((i % 256) as f32 - 128.0) / 100.0).collect();
    let mut q8_blocks =
        vec![BlockQ8_K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; n_blocks];
    infer_train::backend::arm::quant::quantize_q8_k(&input_f32, &mut q8_blocks);

    // Test 1: Standard linear_q8_k
    let mut output_std = vec![0.0f32; out_features];
    linear::linear_q8_k(
        &q8_blocks,
        in_features,
        weight_data,
        tensor.quant_type,
        stride,
        &mut output_std,
        1,
    )
    .expect("linear_q8_k failed");

    // Test 2: Repack + interleaved GEMV
    let size_block_q4_kx8 = std::mem::size_of::<BlockQ4_Kx8>();
    let n_row_groups = out_features / 8;

    let mut repacked_data =
        vec![0u8; n_row_groups * n_blocks * size_block_q4_kx8];
    repack::repack_q4_k_to_x8(
        weight_data,
        stride,
        &mut repacked_data,
        out_features,
        n_blocks,
    );

    let mut output_gemv = vec![0.0f32; out_features];
    linear::linear_q8_k_batch_repacked(
        &q8_blocks,
        &repacked_data,
        tensor.quant_type,
        &mut output_gemv,
        out_features,
        in_features,
        1,
    )
    .expect("linear_q8_k_batch_repacked failed");

    // Compare results
    println!("\n=== Comparison (first 16 rows) ===");
    let mut max_diff = 0.0f32;
    for i in 0..std::cmp::min(16, out_features) {
        let diff = (output_std[i] - output_gemv[i]).abs();
        println!(
            "  row {:3}: std={:10.4}, gemv={:10.4}, diff={:10.4}",
            i, output_std[i], output_gemv[i], diff
        );
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("\nMax diff: {}", max_diff);

    // Check if results match
    let tolerance = 1e-3;
    let mut matches = 0;
    for i in 0..out_features {
        if (output_std[i] - output_gemv[i]).abs() < tolerance {
            matches += 1;
        }
    }
    println!("Matches: {}/{}", matches, out_features);

    assert!(matches > out_features * 9 / 10, "Too many mismatches");
}
