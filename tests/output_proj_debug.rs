use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use infer_train::backend::quant;
use infer_train::model::gguf::GgufModel;
use infer_train::types::{BlockQ6_K, BlockQ8_K, QK_K, QuantType};
use std::fs::File;
use std::io::Read;

fn main() {
    let model_path =
        std::env::args().nth(1).expect("Usage: output_proj_debug <model.gguf>");

    // Load model
    let mut model = GgufModel::load(&model_path).expect("Failed to load model");

    // Get token_embd.weight tensor info
    let tensor = model
        .get_tensor_meta("token_embd.weight")
        .expect("token_embd.weight not found");

    println!("token_embd.weight:");
    println!("  dims: {:?}", tensor.dims);
    println!("  quant_type: {:?}", tensor.quant_type);
    println!("  strides: {:?}", tensor.strides);
    println!("  offset: {}", tensor.offset);

    let hidden_size = tensor.dims[0] as usize;
    let vocab_size = tensor.dims[1] as usize;
    let stride_0 = tensor.strides[0] as usize;
    let stride_1 = tensor.strides[1] as usize;
    let nb = (hidden_size + QK_K - 1) / QK_K;

    // Get tensor data
    let tensor_data = model
        .get_tensor("token_embd.weight")
        .expect("Failed to get tensor data");

    // Create a simple hidden state
    let mut hidden = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        hidden[i] = 0.01 * ((i % 100) as f32);
    }

    // Quantize hidden to Q8_K
    let mut hidden_q8k = vec![
        BlockQ8_K {
            d: 0,
            qs: [0; QK_K],
            s: [0; QK_K / 16],
            bsums: [0; QK_K / 32]
        };
        nb
    ];
    quant::quantize_q8_k(&hidden, &mut hidden_q8k);

    // Compute logits for first 5 tokens
    println!("\nComputing logits for first 5 tokens:");
    for j in 0..5 {
        // Collect blocks for column j
        let mut col_blocks: Vec<BlockQ6_K> =
            vec![unsafe { std::mem::zeroed() }; nb];
        for b in 0..nb {
            let offset = b * stride_0 + j * stride_1;
            col_blocks[b] = unsafe {
                *(tensor_data.as_ptr().add(offset) as *const BlockQ6_K)
            };
        }

        // Compute dot product
        let logits = vec_dot_q6_k_q8_k(&col_blocks, &hidden_q8k);
        println!("  token {}: logits = {:.6}", j, logits);
    }

    // Now let's verify by computing manually
    println!("\nManual verification:");

    // Dequantize column 0 and compute dot product
    // For Q6_K, we need to properly dequantize each block
    // This is complex, so let's just print the first block of column 0

    let block0_offset = 0 * stride_0 + 0 * stride_1;
    let block0: &BlockQ6_K = unsafe {
        &*(tensor_data.as_ptr().add(block0_offset) as *const BlockQ6_K)
    };

    println!("\nBlock 0 of column 0:");
    println!("  ql[0..8]: {:?}", &block0.ql[..8]);
    println!("  qh[0..8]: {:?}", &block0.qh[..8]);
    println!("  scales[0..8]: {:?}", &block0.scales[..8]);
}
