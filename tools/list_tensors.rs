// End-to-end comparison with llama.cpp
// First, let's just run a simple embedding lookup and compare

use infer_train::backend::arm::dequant::dequantize_q6_k_block;
use infer_train::model::GgufFile;
use infer_train::types::BlockQ6_K;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Check the first few bytes of embedding data
    let emb_data =
        gguf.get_tensor_data("token_embd.weight").expect("No embedding data");

    println!("First 32 bytes of embedding data:");
    for i in 0..32 {
        print!("{:02X} ", emb_data[i]);
    }
    println!();

    // Parse as BlockQ6_K
    let block: &BlockQ6_K =
        unsafe { &*(emb_data.as_ptr() as *const BlockQ6_K) };

    println!("\nFirst BlockQ6_K:");
    println!("  d: {} (0x{:04X})", block.d, block.d);
    println!("  ql[0..8]: {:?}", &block.ql[0..8]);
    println!("  qh[0..8]: {:?}", &block.qh[0..8]);
    println!("  scales[0..8]: {:?}", &block.scales[0..8]);

    // Dequantize
    let vals = dequantize_q6_k_block(block);
    println!("\nDequantized values (first 16):");
    for i in 0..16 {
        println!("  [{}]: {:.6}", i, vals[i]);
    }

    // Also check the scales encoding
    // Q6_K scales are i8, but they should be treated as signed
    println!("\nScales interpretation:");
    for i in 0..8 {
        let scale = block.scales[i];
        println!(
            "  scales[{}] = {} (raw byte: 0x{:02X})",
            i, scale, scale as u8
        );
    }

    // Check if scales are reasonable (should be small integers like 1, 2, 3, etc.)
    let scale_sum: i32 = block.scales[0..16].iter().map(|&s| s as i32).sum();
    println!("\nSum of scales: {}", scale_sum);

    // The issue might be with scales being 0 or negative
    // Let's check if scales are being applied correctly
    let d = half::f16::from_bits(block.d).to_f32();
    println!("Super-block scale d: {}", d);

    // For each 16-value sub-block, the actual scale is d * scales[is]
    // Let's check the first sub-block
    println!("\nFirst sub-block (16 values):");
    println!("  Sub-block scale: scales[0] = {}", block.scales[0]);
    println!(
        "  Effective scale: d * scales[0] = {} * {} = {}",
        d,
        block.scales[0],
        d * block.scales[0] as f32
    );

    // Now let's verify the dequantization logic matches llama.cpp
    // From llama.cpp:
    // for (int j = 0; j < QK_K/128; ++j) {
    //     const uint8_t * ql = x[i].ql + j*64;
    //     const uint8_t * qh = x[i].qh + j*32;
    //     const int8_t  * sc = x[i].scales + j*8;
    //     ...
    //     q1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4);
    //     q2 = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
    //     q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
    //     q4 = (ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4);
    // }

    // Let's manually compute the first 4 values correctly
    // For l=0: vals[0] uses ql[0], vals[32] uses ql[32]
    // For l=1: vals[1] uses ql[1], vals[33] uses ql[33]
    // etc.

    let ql_0 = block.ql[0];
    let qh_0 = block.qh[0];
    let ql_1 = block.ql[1];
    let qh_1 = block.qh[1];

    // For l=0:
    let q1_0 = ((ql_0 & 0xF) as i32) | (((qh_0 & 3) as i32) << 4);
    let q2_0 =
        ((block.ql[32] & 0xF) as i32) | ((((qh_0 >> 2) & 3) as i32) << 4);
    let q3_0 = ((ql_0 >> 4) as i32) | ((((qh_0 >> 4) & 3) as i32) << 4);
    let q4_0 = ((block.ql[32] >> 4) as i32) | ((((qh_0 >> 6) & 3) as i32) << 4);

    // For l=1:
    let q1_1 = ((ql_1 & 0xF) as i32) | (((qh_1 & 3) as i32) << 4);

    println!("\nManual dequant for l=0:");
    println!(
        "  ql[0]=0x{:02X}, ql[32]=0x{:02X}, qh[0]=0x{:02X}",
        ql_0, block.ql[32], qh_0
    );
    println!("  q1 (raw) = {}, q1 - 32 = {}", q1_0, q1_0 - 32);
    println!("  vals[0] should be: d * scales[0] * (q1 - 32)");
    println!(
        "  vals[0] = {} * {} * {} = {}",
        d,
        block.scales[0],
        q1_0 - 32,
        d * block.scales[0] as f32 * (q1_0 - 32) as f32
    );

    println!("\nManual dequant for l=1:");
    println!("  ql[1]=0x{:02X}, qh[1]=0x{:02X}", ql_1, qh_1);
    println!("  q1 (raw) = {}, q1 - 32 = {}", q1_1, q1_1 - 32);
    println!("  vals[1] should be: d * scales[0] * (q1 - 32)");
    println!(
        "  vals[1] = {} * {} * {} = {}",
        d,
        block.scales[0],
        q1_1 - 32,
        d * block.scales[0] as f32 * (q1_1 - 32) as f32
    );
}
