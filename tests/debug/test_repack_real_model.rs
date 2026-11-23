// Test repack with real model weights
use infer_train::model::GgufFile;
use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ6_K, BlockQ6_Kx8, BlockQ8_K, QK_K};
use infer_train::backend::arm::repack::repack_q5_k_to_x8;
use infer_train::backend::arm::repack::repack_q6_k_to_x8;
use infer_train::backend::arm::vec_dot::vec_dot_q5_k_q8_k;
use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use infer_train::backend::arm::gemv::gemv_q5_k_x8_neon;
use infer_train::backend::arm::gemv::gemv_q6_k_x8_neon;

fn main() {
    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load model");

    // Find a Q5_K weight tensor (attn_q, attn_k, attn_v, etc.)
    let tensor = gguf.tensors.iter()
        .find(|t| t.name.contains("attn_q.weight"))
        .or_else(|| gguf.tensors.iter().find(|t| t.name.contains("attn_k.weight")))
        .or_else(|| gguf.tensors.iter().find(|t| t.name.contains("ffn_gate.weight")))
        .or_else(|| gguf.tensors.iter().find(|t| t.name.contains("ffn_up.weight")))
        .or_else(|| gguf.tensors.iter().find(|t| t.name.contains("ffn_down.weight")));

    let tensor = match tensor {
        Some(t) => t,
        None => {
            // Find any Q5_K tensor
            let t = gguf.tensors.iter()
                .find(|t| t.quant_type == infer_train::types::QuantType::Q5_K);
            match t {
                Some(t) => t,
                None => {
                    println!("No Q5_K tensor found, trying Q6_K");
                    // Try Q6_K
                    let t = gguf.tensors.iter()
                        .find(|t| t.quant_type == infer_train::types::QuantType::Q6_K);
                    match t {
                        Some(t) => test_q6_k(&gguf, t),
                        None => {
                            println!("No Q5_K or Q6_K tensor found");
                            return;
                        }
                    }
                    return;
                }
            }
        }
    };

    test_q5_k(&gguf, tensor);
}

fn test_q5_k(gguf: &GgufFile, tensor: &infer_train::model::GgufTensorInfo) {
    println!("=== Testing Q5_K tensor: {} ===", tensor.name);
    println!("Dims: {:?}", tensor.dims);
    println!("Strides: {:?}", tensor.strides);
    println!("Quant type: {:?}", tensor.quant_type);

    let weight_data = gguf.get_tensor_data(&tensor.name).expect("Failed to get tensor data");

    let ne0 = tensor.dims.get(0).copied().unwrap_or(0) as usize; // in_features
    let ne1 = tensor.dims.get(1).copied().unwrap_or(0) as usize; // out_features
    let nb0 = tensor.strides.get(0).copied().unwrap_or(0) as usize;
    let nb1 = tensor.strides.get(1).copied().unwrap_or(0) as usize;

    let size_block_q5_k = std::mem::size_of::<BlockQ5_K>();
    let blocks_per_row = ne0 / QK_K;

    println!("\nLayout analysis:");
    println!("  in_features (ne0) = {}", ne0);
    println!("  out_features (ne1) = {}", ne1);
    println!("  blocks_per_row = {}", blocks_per_row);
    println!("  nb0 = {} (expected {})", nb0, size_block_q5_k);
    println!("  nb1 = {}", nb1);
    println!("  interleave_factor = {}", nb1 / size_block_q5_k);

    // Check if we can test with 8 rows
    if ne1 < 8 {
        println!("Tensor has only {} rows, need at least 8 for interleaved GEMV test", ne1);
        return;
    }

    // Read first 8 rows using stride
    // For GGUF interleaved format with factor=4:
    // - Rows 0-3 are interleaved together
    // - Rows 4-7 are interleaved together
    // Each row has `blocks_per_row` blocks

    println!("\nReading first 8 rows with blocks_per_row = {}...", blocks_per_row);

    let mut src_blocks: Vec<Vec<BlockQ5_K>> = Vec::new();
    for row in 0..8 {
        let row_offset = row * nb1;
        let row_blocks: Vec<BlockQ5_K> = (0..blocks_per_row)
            .map(|blk| {
                let offset = row_offset + blk * size_block_q5_k;
                if offset + size_block_q5_k > weight_data.len() {
                    println!("WARNING: offset {} out of bounds", offset);
                    unsafe { std::mem::zeroed() }
                } else {
                    unsafe {
                        let ptr = weight_data.as_ptr().add(offset) as *const BlockQ5_K;
                        std::ptr::read(ptr)
                    }
                }
            })
            .collect();
        src_blocks.push(row_blocks);
    }

    // Extract just the first block of each row for single-block comparison
    let first_blocks: Vec<BlockQ5_K> = src_blocks.iter().map(|r| r[0]).collect();

    println!("\nFirst 8 rows, first block each:");
    for (i, blk) in first_blocks.iter().enumerate() {
        let d = half::f16::from_bits(blk.d).to_f32();
        let dmin = half::f16::from_bits(blk.dmin).to_f32();
        println!("Row {}: d={:.4}, dmin={:.4}, scales={:?}", i, d, dmin, &blk.scales[..4]);
    }

    // Create Q8_K input (all 1s for simplicity)
    let mut q8_blocks: Vec<BlockQ8_K> = (0..blocks_per_row)
        .map(|_| {
            let mut blk: BlockQ8_K = unsafe { std::mem::zeroed() };
            blk.d = 1.0;
            blk.qs.fill(1);
            blk
        })
        .collect();

    // Test standard vec_dot for each row (using all blocks)
    println!("\n=== Standard vec_dot (all blocks) ===");
    let mut std_results = [0.0f32; 8];
    for i in 0..8 {
        std_results[i] = vec_dot_q5_k_q8_k(&src_blocks[i], &q8_blocks);
        println!("Row {}: {:.4}", i, std_results[i]);
    }

    // Repack 8 rows - need to flatten all blocks
    let src_bytes: Vec<u8> = src_blocks.iter()
        .flat_map(|row| {
            row.iter().flat_map(|b| unsafe {
                std::slice::from_raw_parts(b as *const _ as *const u8, size_block_q5_k).iter().copied()
            })
        })
        .collect();

    let size_block_q5_kx8 = std::mem::size_of::<BlockQ5_Kx8>();
    let n_row_groups = 8 / 8; // 1 group of 8 rows
    let dst_size = n_row_groups * blocks_per_row * size_block_q5_kx8;
    let mut dst_bytes = vec![0u8; dst_size];

    // src_stride should be the distance between rows in source layout
    // For GGUF interleaved: nb1 = 4 * size_block (not blocks_per_row * size_block)
    let src_stride = nb1;

    println!("\n=== Repacking ===");
    println!("src_stride = {} (interleave_factor = {})", src_stride, src_stride / size_block_q5_k);
    println!("n_rows = 8, n_blocks_per_row = {}", blocks_per_row);
    println!("is_gguf_interleaved check: interleave_factor == 4 && blocks_per_row({}) > 4 = {}",
             blocks_per_row, (src_stride / size_block_q5_k) == 4 && blocks_per_row > 4);

    repack_q5_k_to_x8(&src_bytes, src_stride, &mut dst_bytes, 8, blocks_per_row);

    // Interleaved GEMV
    println!("\n=== Interleaved GEMV ===");
    let mut gemv_results = [0.0f32; 8];
    unsafe {
        let weights_x8: &[BlockQ5_Kx8] = std::slice::from_raw_parts(
            dst_bytes.as_ptr() as *const BlockQ5_Kx8,
            n_row_groups * blocks_per_row
        );
        gemv_q5_k_x8_neon(weights_x8, &q8_blocks, &mut gemv_results, 8, ne0);
    }
    println!("GEMV results: {:?}", gemv_results);

    // Compare
    println!("\n=== Comparison ===");
    for i in 0..8 {
        let diff = (std_results[i] - gemv_results[i]).abs();
        let status = if diff < 0.01 { "PASS" } else { "FAIL" };
        println!("Row {}: std={:.4}, gemv={:.4}, diff={:.4} [{}]",
                 i, std_results[i], gemv_results[i], diff, status);
    }
}

fn test_q6_k(gguf: &GgufFile, tensor: &infer_train::model::GgufTensorInfo) {
    println!("=== Testing Q6_K tensor: {} ===", tensor.name);
    println!("Dims: {:?}", tensor.dims);
    println!("Strides: {:?}", tensor.strides);

    let weight_data = gguf.get_tensor_data(&tensor.name).expect("Failed to get tensor data");

    let ne0 = tensor.dims.get(0).copied().unwrap_or(0) as usize;
    let ne1 = tensor.dims.get(1).copied().unwrap_or(0) as usize;
    let nb0 = tensor.strides.get(0).copied().unwrap_or(0) as usize;
    let nb1 = tensor.strides.get(1).copied().unwrap_or(0) as usize;

    let size_block_q6_k = std::mem::size_of::<BlockQ6_K>();
    let blocks_per_row = ne0 / QK_K;

    println!("\nLayout analysis:");
    println!("  in_features = {}, out_features = {}", ne0, ne1);
    println!("  blocks_per_row = {}", blocks_per_row);
    println!("  nb0 = {} (expected {})", nb0, size_block_q6_k);
    println!("  nb1 = {}", nb1);
    println!("  interleave_factor = {}", nb1 / size_block_q6_k);

    if ne1 < 8 {
        println!("Tensor has only {} rows, need at least 8", ne1);
        return;
    }

    // Read first 8 rows
    let mut src_blocks: Vec<BlockQ6_K> = Vec::new();
    for row in 0..8 {
        let row_offset = row * nb1;
        let offset = row_offset; // first block
        if offset + size_block_q6_k > weight_data.len() {
            src_blocks.push(unsafe { std::mem::zeroed() });
        } else {
            unsafe {
                let ptr = weight_data.as_ptr().add(offset) as *const BlockQ6_K;
                src_blocks.push(std::ptr::read(ptr));
            }
        }
    }

    println!("\nFirst 8 rows:");
    for (i, blk) in src_blocks.iter().enumerate() {
        let d = half::f16::from_bits(blk.d).to_f32();
        println!("Row {}: d={:.4}, scales={:?}", i, d, &blk.scales[..4]);
    }

    // Create Q8_K input
    let mut q8_block: BlockQ8_K = unsafe { std::mem::zeroed() };
    q8_block.d = 1.0;
    q8_block.qs.fill(1);

    // Standard vec_dot
    println!("\n=== Standard vec_dot ===");
    let mut std_results = [0.0f32; 8];
    for i in 0..8 {
        std_results[i] = vec_dot_q6_k_q8_k(&[src_blocks[i]], &[q8_block]);
        println!("Row {}: {:.4}", i, std_results[i]);
    }

    // Repack
    let src_bytes: Vec<u8> = src_blocks.iter()
        .flat_map(|b| unsafe {
            std::slice::from_raw_parts(b as *const _ as *const u8, size_block_q6_k).iter().copied()
        })
        .collect();

    let size_block_q6_kx8 = std::mem::size_of::<BlockQ6_Kx8>();
    let mut dst_bytes = vec![0u8; size_block_q6_kx8];

    repack_q6_k_to_x8(&src_bytes, nb1, &mut dst_bytes, 8, 1);

    // Interleaved GEMV
    println!("\n=== Interleaved GEMV ===");
    let mut gemv_results = [0.0f32; 8];
    unsafe {
        let dst_block = &*(dst_bytes.as_ptr() as *const BlockQ6_Kx8);
        gemv_q6_k_x8_neon(&[dst_block.clone()], &[q8_block], &mut gemv_results, 8, 256);
    }
    println!("GEMV results: {:?}", gemv_results);

    // Compare
    println!("\n=== Comparison ===");
    for i in 0..8 {
        let diff = (std_results[i] - gemv_results[i]).abs();
        let status = if diff < 0.01 { "PASS" } else { "FAIL" };
        println!("Row {}: std={:.4}, gemv={:.4}, diff={:.4} [{}]",
                 i, std_results[i], gemv_results[i], diff, status);
    }
}
