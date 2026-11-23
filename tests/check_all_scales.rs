// Check scales for all first layer weights
use infer_train::model::GgufFile;
use infer_train::types::{BlockQ5_K, BlockQ6_K};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    let tensors = [
        ("blk.0.attn_q.weight", "Q5_K"),
        ("blk.0.attn_k.weight", "Q5_K"),
        ("blk.0.attn_v.weight", "Q6_K"),
        ("blk.0.ffn_gate.weight", "Q5_K"),
        ("blk.0.ffn_up.weight", "Q5_K"),
        ("blk.0.ffn_down.weight", "Q6_K"),
    ];

    for (name, expected_type) in tensors {
        let tensor = gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .expect(&format!("{} not found", name));
        let data = gguf
            .get_tensor_data(name)
            .expect(&format!("{} data not found", name));

        println!("\n=== {} ===", name);
        println!("dims: {:?}, strides: {:?}", tensor.dims, tensor.strides);
        println!("quant: {:?}", tensor.quant_type);

        let nb = (tensor.dims[0] as usize + 255) / 256;

        // Read first row
        if tensor.quant_type == infer_train::types::QuantType::Q5_K {
            let block_size = std::mem::size_of::<BlockQ5_K>();
            let row_stride = nb * block_size;

            // Sample 5 rows
            println!("Scales (d, dmin) for rows 0, 100, 500, 1000:");
            for row in [0, 100, 500, 1000] {
                if row < tensor.dims[1] as usize {
                    let offset = row * row_stride;
                    let block: &BlockQ5_K = unsafe {
                        &*(data.as_ptr().add(offset) as *const BlockQ5_K)
                    };
                    let d = half::f16::from_bits(block.d).to_f32();
                    let dmin = half::f16::from_bits(block.dmin).to_f32();
                    println!("  Row {}: d={:.6}, dmin={:.6}", row, d, dmin);
                }
            }
        } else if tensor.quant_type == infer_train::types::QuantType::Q6_K {
            let block_size = std::mem::size_of::<BlockQ6_K>();
            let row_stride = nb * block_size;

            // Sample 5 rows
            println!("Scales (d) for rows 0, 100, 500, 1000:");
            for row in [0, 100, 500, 1000] {
                if row < tensor.dims[1] as usize {
                    let offset = row * row_stride;
                    let block: &BlockQ6_K = unsafe {
                        &*(data.as_ptr().add(offset) as *const BlockQ6_K)
                    };
                    let d = half::f16::from_bits(block.d).to_f32();
                    let scales_min =
                        block.scales.iter().min().copied().unwrap();
                    let scales_max =
                        block.scales.iter().max().copied().unwrap();
                    println!(
                        "  Row {}: d={:.6}, scales=[{}, {}]",
                        row, d, scales_min, scales_max
                    );
                }
            }
        }
    }
}
