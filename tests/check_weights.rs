// Check actual Q5_K weights from the model
use infer_train::backend::arm::vec_dot;
use infer_train::model::GgufFile;
use infer_train::types::{BlockQ5_K, BlockQ8_K, QK_K};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Get Q projection weights
    let q_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("Q tensor not found");
    let q_data =
        gguf.get_tensor_data("blk.0.attn_q.weight").expect("Q data not found");

    println!(
        "Q tensor: dims={:?}, quant={:?}",
        q_tensor.dims, q_tensor.quant_type
    );
    println!("Q data length: {} bytes", q_data.len());

    // Expected: dims=[1024, 2048], so 1024 in_features, 2048 out_features
    // Each row has (1024/256) = 4 blocks
    // Total blocks per row = 4
    // Block size = 176 bytes
    // Row stride = 4 * 176 = 704 bytes
    // Total data = 2048 * 704 = 1,441,792 bytes

    let nb = 4; // blocks per row
    let block_size = std::mem::size_of::<BlockQ5_K>();
    println!("Block size: {} bytes", block_size);

    // Read first row
    let first_block: &BlockQ5_K =
        unsafe { &*(q_data.as_ptr() as *const BlockQ5_K) };

    println!("\nFirst block of first row:");
    println!(
        "  d = {:.6} (bits = {})",
        half::f16::from_bits(first_block.d).to_f32(),
        first_block.d
    );
    println!(
        "  dmin = {:.6} (bits = {})",
        half::f16::from_bits(first_block.dmin).to_f32(),
        first_block.dmin
    );
    println!("  scales[:8] = {:?}", &first_block.scales[..8]);
    println!("  qs[:8] = {:?}", &first_block.qs[..8]);
    println!("  qh[:8] = {:?}", &first_block.qh[..8]);

    // Read all blocks of first row
    let row_stride = nb * block_size;
    let first_row_blocks: Vec<BlockQ5_K> = (0..nb)
        .map(|j| unsafe {
            let ptr = q_data.as_ptr().add(j * block_size) as *const BlockQ5_K;
            (*ptr).clone()
        })
        .collect();

    println!("\nFirst row (4 blocks):");
    for (i, blk) in first_row_blocks.iter().enumerate() {
        println!(
            "  Block {}: d={:.4}, dmin={:.4}",
            i,
            half::f16::from_bits(blk.d).to_f32(),
            half::f16::from_bits(blk.dmin).to_f32()
        );
    }

    // Create a simple input
    let mut input: Vec<BlockQ8_K> = Vec::new();
    for _ in 0..nb {
        let mut block: BlockQ8_K = unsafe { std::mem::zeroed() };
        block.d = 1.0;
        for i in 0..256 {
            block.qs[i] = 1; // All ones
        }
        // Compute bsums
        for s in 0..16 {
            let mut sum: i16 = 0;
            for k in 0..16 {
                sum += block.qs[s * 16 + k] as i16;
            }
            block.bsums[s] = sum;
        }
        input.push(block);
    }

    // Compute dot product for first row
    let result =
        unsafe { vec_dot::vec_dot_q5_k_q8_k(&first_row_blocks, &input) };
    println!("\nDot product with all-ones input: {}", result);

    // Now check the embedding weights
    let emb_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("Embedding tensor not found");
    let emb_data = gguf
        .get_tensor_data("token_embd.weight")
        .expect("Embedding data not found");

    println!(
        "\nEmbedding tensor: dims={:?}, quant={:?}",
        emb_tensor.dims, emb_tensor.quant_type
    );
    println!("Embedding data length: {} bytes", emb_data.len());

    // Read first embedding (token 0)
    // Embedding is stored as [hidden_size, vocab_size]
    // Each column is an embedding vector
    // For F16: stride[1] should be 2 (2 bytes per element)

    if emb_tensor.quant_type == infer_train::types::QuantType::Q6_K {
        println!("Embedding uses Q6_K quantization");
        let nb_emb = (emb_tensor.dims[0] as usize + QK_K - 1) / QK_K;
        println!("  nb blocks per embedding: {}", nb_emb);
    }
}
