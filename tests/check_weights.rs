//! Check weight values in model
use half::f16;
use infer_train::model::gguf::GgufFile;
use infer_train::types::BlockQ6_K;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let gguf = GgufFile::load(&args[1]).expect("Failed to open GGUF");

    // Find token_embd.weight tensor
    let embd_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight not found");

    println!("Embedding tensor:");
    println!("  name: {}", embd_tensor.name);
    println!("  dims: {:?}", embd_tensor.dims);
    println!("  strides: {:?}", embd_tensor.strides);
    println!("  quant_type: {:?}", embd_tensor.quant_type);

    // Load tensor data
    let data = gguf
        .get_tensor_data(&embd_tensor.name)
        .expect("Failed to get tensor data");

    // Check first few blocks
    let blocks: &[BlockQ6_K] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const BlockQ6_K,
            data.len() / std::mem::size_of::<BlockQ6_K>(),
        )
    };

    println!("\nFirst 10 blocks:");
    for i in 0..10.min(blocks.len()) {
        let block = &blocks[i];
        let d = f16::from_bits(block.d).to_f32();
        println!(
            "  Block {}: d={:.8}, scales[0..4]={:?}",
            i,
            d,
            &block.scales[0..4]
        );
    }

    // Check a few specific tokens
    println!("\nSpecific tokens (assuming column-major layout):");
    let hidden_size = 1024usize;
    let blocks_per_token = hidden_size / 256;
    let block_size = std::mem::size_of::<BlockQ6_K>();

    for token_id in [0, 9707, 151644] {
        let token_offset = token_id * blocks_per_token * block_size;
        let token_blocks: &[BlockQ6_K] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().add(token_offset) as *const BlockQ6_K,
                blocks_per_token,
            )
        };
        let d = f16::from_bits(token_blocks[0].d).to_f32();
        println!(
            "  Token {}: first block d={:.8}, scales={:?}",
            token_id,
            d,
            &token_blocks[0].scales[0..4]
        );
    }
}
