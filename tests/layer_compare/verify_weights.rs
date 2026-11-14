use infer_train::model::GgufFile;
use infer_train::types::*;

fn main() {
    let path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(path).unwrap();

    // Check token embedding
    if let Some(emb) =
        gguf.tensors.iter().find(|t| t.name == "token_embd.weight")
    {
        println!("token_embd: {:?}", emb);
        let data = gguf.get_tensor_data("token_embd.weight").unwrap();
        let nb = 1024 / 256; // blocks per row
        let block_bytes = std::mem::size_of::<BlockQ6_K>();

        // Check first few blocks
        for i in 0..3.min(nb) {
            let offset = i * block_bytes;
            let block_bytes_slice = &data[offset..offset + block_bytes];
            let block: &BlockQ6_K =
                unsafe { &*(block_bytes_slice.as_ptr() as *const BlockQ6_K) };
            println!(
                "Block {}: d={:.4} ql[0:4]={:?} scales[0:4]={:?}",
                i,
                fp16_to_fp32(block.d),
                &block.ql[0..4],
                &block.scales[0..4]
            );
        }
    }

    // Check attn_q weight
    if let Some(q) =
        gguf.tensors.iter().find(|t| t.name == "blk.0.attn_q.weight")
    {
        println!("\nattn_q: {:?}", q);
        let data = gguf.get_tensor_data("blk.0.attn_q.weight").unwrap();
        let nb = 1024 / 256; // blocks per row for in_features=1024
        let blocks_per_row = nb;
        let block_bytes = std::mem::size_of::<BlockQ5_K>();

        // Check first row's first few blocks
        println!("First row blocks:");
        for i in 0..2.min(blocks_per_row) {
            let offset = i * block_bytes;
            let block_bytes_slice = &data[offset..offset + block_bytes];
            let block: &BlockQ5_K =
                unsafe { &*(block_bytes_slice.as_ptr() as *const BlockQ5_K) };
            println!(
                "  Block {}: d={:.4} dmin={:.4} scales[0:4]={:?} qs[0:4]={:?}",
                i,
                fp16_to_fp32(block.d),
                fp16_to_fp32(block.dmin),
                &block.scales[0..4],
                &block.qs[0..4]
            );
        }
    }
}

fn fp16_to_fp32(h: u16) -> f32 {
    half::f16::from_bits(h).to_f32()
}
