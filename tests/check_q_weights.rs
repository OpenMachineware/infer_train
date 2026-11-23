// Debug weight loading for Q5_K
use infer_train::model::GgufFile;
use infer_train::types::BlockQ5_K;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    let q_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("Q tensor not found");
    let q_data =
        gguf.get_tensor_data("blk.0.attn_q.weight").expect("Q data not found");

    println!("Q tensor:");
    println!("  dims = {:?}", q_tensor.dims);
    println!("  strides = {:?}", q_tensor.strides);
    println!("  quant_type = {:?}", q_tensor.quant_type);
    println!("  data length = {} bytes", q_data.len());

    let nb = (q_tensor.dims[0] as usize + 255) / 256; // blocks per row
    let block_size = std::mem::size_of::<BlockQ5_K>();
    let expected_stride = nb * block_size;
    let expected_total = q_tensor.dims[1] as usize * expected_stride;

    println!("\nExpected:");
    println!("  blocks per row: {}", nb);
    println!("  block size: {} bytes", block_size);
    println!("  expected row stride: {} bytes", expected_stride);
    println!("  expected total: {} bytes", expected_total);

    // Check stored stride
    if q_tensor.strides.len() > 1 {
        let stored_stride = q_tensor.strides[1] as usize;
        println!("  stored stride: {} bytes", stored_stride);
        if stored_stride != expected_stride {
            println!("  WARNING: stride mismatch!");
        }
    }

    // Read first few rows and print their scales
    println!("\nFirst 5 rows' scales:");
    for row in 0..5 {
        let row_offset = row * expected_stride;
        let first_block: &BlockQ5_K =
            unsafe { &*(q_data.as_ptr().add(row_offset) as *const BlockQ5_K) };

        let d = half::f16::from_bits(first_block.d).to_f32();
        let dmin = half::f16::from_bits(first_block.dmin).to_f32();
        println!("  Row {}: d={:.6}, dmin={:.6}", row, d, dmin);
    }

    // Check if data is stored in a different order
    // GGUF stores dims as [in_features, out_features]
    // But the actual storage could be column-major or row-major

    // Let's check if the first block of each column has reasonable scales
    println!("\nFirst block of columns 0, 100, 500, 1000, 2000:");
    for col in [0, 100, 500, 1000, 2000] {
        if col < q_tensor.dims[1] as usize {
            // If column-major: stride[0] would be the size of one element
            // If row-major: stride[1] would be the size of one row
            let col_offset = col * expected_stride; // Assuming row-major
            let first_block: &BlockQ5_K = unsafe {
                &*(q_data.as_ptr().add(col_offset) as *const BlockQ5_K)
            };

            let d = half::f16::from_bits(first_block.d).to_f32();
            let dmin = half::f16::from_bits(first_block.dmin).to_f32();
            println!("  Col {}: d={:.6}, dmin={:.6}", col, d, dmin);
        }
    }

    // Let's also check the raw bytes
    println!("\nFirst 32 bytes (hex):");
    for i in 0..32 {
        print!("{:02x} ", q_data[i]);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }

    // BlockQ5_K layout:
    // offset 0-1: d (u16)
    // offset 2-3: dmin (u16)
    // offset 4-15: scales (12 bytes)
    // offset 16-47: qh (32 bytes)
    // offset 48-175: qs (128 bytes)

    println!("\nInterpreting first block:");
    let d_bytes = [q_data[0], q_data[1]];
    let dmin_bytes = [q_data[2], q_data[3]];
    let d_val = half::f16::from_le_bytes(d_bytes).to_f32();
    let dmin_val = half::f16::from_le_bytes(dmin_bytes).to_f32();
    println!(
        "  d = {:.6} (bytes: {:02x} {:02x})",
        d_val, d_bytes[0], d_bytes[1]
    );
    println!(
        "  dmin = {:.6} (bytes: {:02x} {:02x})",
        dmin_val, dmin_bytes[0], dmin_bytes[1]
    );
}
