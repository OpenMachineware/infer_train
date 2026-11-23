// Debug Q6_K scales in actual model
use infer_train::model::GgufFile;
use infer_train::types::{BlockQ5_K, BlockQ6_K};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Check Q weight which is Q5_K
    let q_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("Q weight not found");
    let q_data =
        gguf.get_tensor_data("blk.0.attn_q.weight").expect("q data not found");

    println!("=== Q weight (Q5_K) ===");
    println!("dims={:?}, quant={:?}", q_tensor.dims, q_tensor.quant_type);

    let nb = (q_tensor.dims[0] as usize + 255) / 256;
    let block_size = std::mem::size_of::<BlockQ5_K>();
    let row_stride = nb * block_size;

    println!("nb={}, block_size={}, row_stride={}", nb, block_size, row_stride);

    // Check first few rows
    for row in 0..5 {
        let offset = row * row_stride;
        let block: &BlockQ5_K =
            unsafe { &*(q_data.as_ptr().add(offset) as *const BlockQ5_K) };

        let d = half::f16::from_bits(block.d).to_f32();
        let dmin = half::f16::from_bits(block.dmin).to_f32();

        println!("\nRow {}: d={:.6}, dmin={:.6}", row, d, dmin);

        // Decode scales and mins from 12-byte format
        // This is complex 6-bit encoding
        // Just print raw bytes for now
        println!("  scales raw: {:?}", &block.scales[..12]);

        // Check scales encoding
        // K_SCALE_SIZE = 12, encodes 8 scales + 8 mins
        // Each scale/min is 6 bits
        let mut utmp = [0u32; 4];
        unsafe {
            std::ptr::copy_nonoverlapping(
                block.scales.as_ptr(),
                utmp.as_mut_ptr() as *mut u8,
                12,
            );
        }

        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let scales: [u8; 8] = [
            (utmp[0] & 0x3f) as u8,
            ((utmp[0] >> 8) & 0x3f) as u8,
            ((utmp[0] >> 16) & 0x3f) as u8,
            ((utmp[0] >> 24) & 0x3f) as u8,
            (utmp[1] & 0x3f) as u8,
            ((utmp[1] >> 8) & 0x3f) as u8,
            ((utmp[1] >> 16) & 0x3f) as u8,
            ((utmp[1] >> 24) & 0x3f) as u8,
        ];

        let mins: [u8; 8] = [
            (utmp[2] & 0x3f) as u8,
            ((utmp[2] >> 8) & 0x3f) as u8,
            ((utmp[2] >> 16) & 0x3f) as u8,
            ((utmp[2] >> 24) & 0x3f) as u8,
            (utmp[3] & 0x3f) as u8,
            ((utmp[3] >> 8) & 0x3f) as u8,
            ((utmp[3] >> 16) & 0x3f) as u8,
            ((utmp[3] >> 24) & 0x3f) as u8,
        ];

        println!("  decoded scales: {:?}", scales);
        println!("  decoded mins: {:?}", mins);
    }

    // Also check V weight which is Q6_K
    let v_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_v.weight")
        .expect("V weight not found");
    let v_data =
        gguf.get_tensor_data("blk.0.attn_v.weight").expect("v data not found");

    println!("\n\n=== V weight (Q6_K) ===");
    println!("dims={:?}, quant={:?}", v_tensor.dims, v_tensor.quant_type);

    let v_nb = (v_tensor.dims[0] as usize + 255) / 256;
    let v_block_size = std::mem::size_of::<BlockQ6_K>();
    let v_row_stride = v_nb * v_block_size;

    println!(
        "nb={}, block_size={}, row_stride={}",
        v_nb, v_block_size, v_row_stride
    );

    // Check first few rows
    for row in 0..3 {
        let offset = row * v_row_stride;
        let block: &BlockQ6_K =
            unsafe { &*(v_data.as_ptr().add(offset) as *const BlockQ6_K) };

        let d = half::f16::from_bits(block.d).to_f32();
        let scales_min = block.scales.iter().min().copied().unwrap();
        let scales_max = block.scales.iter().max().copied().unwrap();

        println!("\nRow {}: d={:.6}", row, d);
        println!("  scales (i8): {:?}", block.scales);
        println!("  scales range: {} to {}", scales_min, scales_max);
    }
}
