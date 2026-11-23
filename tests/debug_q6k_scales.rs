// Debug Q6_K scales in actual model
use infer_train::model::GgufFile;
use infer_train::types::{BlockQ6_K, QuantType};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Check V weight which is Q6_K
    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_v.weight")
        .expect("V weight not found");

    println!("V weight: dims={:?}, quant={:?}", tensor.dims, tensor.quant_type);

    let data =
        gguf.get_tensor_data("blk.0.attn_v.weight").expect("data not found");
    let nb = (tensor.dims[0] as usize + 255) / 256; // blocks per row
    let block_size = std::mem::size_of::<BlockQ6_K>();
    let row_stride = nb * block_size;

    println!("nb={}, block_size={}, row_stride={}", nb, block_size, row_stride);

    // Check first 3 rows
    for row in 0..3 {
        let offset = row * row_stride;
        let block: &BlockQ6_K =
            unsafe { &*(data.as_ptr().add(offset) as *const BlockQ6_K) };

        let d = half::f16::from_bits(block.d).to_f32();
        let scales_min = block.scales.iter().min().copied().unwrap();
        let scales_max = block.scales.iter().max().copied().unwrap();

        println!("\nRow {}: d={:.6}", row, d);
        println!("  scales (i8): {:?}", block.scales);
        println!("  scales range: {} to {}", scales_min, scales_max);

        // Check ql/qh
        let ql_first: Vec<u8> = block.ql[0..16].to_vec();
        let qh_first: Vec<u8> = block.qh[0..8].to_vec();
        println!("  ql[0..16]: {:?}", ql_first);
        println!("  qh[0..8]: {:?}", qh_first);

        // Manually decode first 16 values
        let mut decoded = Vec::new();
        for i in 0..16 {
            let low4 = block.ql[i] & 0x0F;
            let high2 = block.qh[i / 2] >> ((i % 2) * 2);
            let q6 = low4 | ((high2 & 0x03) << 4);
            decoded.push(q6);
        }
        println!("  decoded q6[0..16]: {:?}", decoded);
    }

    // Also check Q weight which is Q5_K for comparison
    let q_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("Q weight not found");
    println!(
        "\n\nQ weight: dims={:?}, quant={:?}",
        q_tensor.dims, q_tensor.quant_type
    );
}
