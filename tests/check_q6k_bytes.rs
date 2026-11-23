// Check raw bytes of Q6_K weights
use infer_train::model::GgufFile;
use infer_train::types::BlockQ6_K;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_v.weight")
        .expect("V tensor not found");
    let data =
        gguf.get_tensor_data("blk.0.attn_v.weight").expect("V data not found");

    println!("V tensor: dims={:?}, strides={:?}", tensor.dims, tensor.strides);

    let nb = (tensor.dims[0] as usize + 255) / 256;
    let block_size = std::mem::size_of::<BlockQ6_K>();
    let row_stride = nb * block_size;

    println!(
        "blocks per row: {}, block size: {}, row stride: {}",
        nb, block_size, row_stride
    );

    // Check first 3 rows
    for row in 0..3 {
        let offset = row * row_stride;
        println!("\n=== Row {} (offset {}) ===", row, offset);

        // Print first 32 bytes
        print!("First 32 bytes: ");
        for i in 0..32 {
            print!("{:02x} ", data[offset + i]);
        }
        println!();

        // Read as BlockQ6_K
        let block: &BlockQ6_K =
            unsafe { &*(data.as_ptr().add(offset) as *const BlockQ6_K) };

        // d is at offset 0-1
        let d_bytes = [data[offset], data[offset + 1]];
        let d_val = half::f16::from_le_bytes(d_bytes).to_f32();
        println!(
            "d bytes: {:02x} {:02x} -> value: {:.6}",
            d_bytes[0], d_bytes[1], d_val
        );

        // scales are at offset 2-17
        print!("scales: ");
        for i in 0..16 {
            print!("{} ", block.scales[i]);
        }
        println!();

        // Print first few ql values
        print!("ql[:8]: ");
        for i in 0..8 {
            print!("{:02x} ", block.ql[i]);
        }
        println!();

        // Print first few qh values
        print!("qh[:8]: ");
        for i in 0..8 {
            print!("{:02x} ", block.qh[i]);
        }
        println!();
    }

    // Compare with Q5_K
    println!("\n=== Compare with Q5_K (attn_q.weight) ===");
    let q_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("Q tensor not found");
    let q_data =
        gguf.get_tensor_data("blk.0.attn_q.weight").expect("Q data not found");

    println!(
        "Q tensor: dims={:?}, strides={:?}",
        q_tensor.dims, q_tensor.strides
    );

    // Print first 32 bytes of Q tensor
    print!("Q first 32 bytes: ");
    for i in 0..32 {
        print!("{:02x} ", q_data[i]);
    }
    println!();

    // Read as BlockQ5_K
    use infer_train::types::BlockQ5_K;
    let q_block: &BlockQ5_K =
        unsafe { &*(q_data.as_ptr() as *const BlockQ5_K) };
    let q_d = half::f16::from_bits(q_block.d).to_f32();
    let q_dmin = half::f16::from_bits(q_block.dmin).to_f32();
    println!("Q d={:.6}, dmin={:.6}", q_d, q_dmin);
}
