use infer_train::model::GgufFile;
use infer_train::types::{BlockQ5_K, BlockQ8_K, QK_K, fp16_to_fp32};
use infer_train::backend::arm::{quant::quantize_q8_k, vec_dot::vec_dot_q5_k_q8_k};

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");
    
    // Get L2 FFN gate weight
    let gate_name = "blk.2.ffn_gate.weight";
    let gate_data = gguf.get_tensor_data(gate_name).expect("No gate weight");
    
    println!("Gate weight data len: {} bytes", gate_data.len());
    
    // Parse Q5_K blocks
    let nb = gate_data.len() / std::mem::size_of::<BlockQ5_K>();
    println!("Number of Q5_K blocks: {}", nb);
    
    let gate_blocks: Vec<BlockQ5_K> = unsafe {
        std::slice::from_raw_parts(gate_data.as_ptr() as *const BlockQ5_K, nb).to_vec()
    };
    
    // Check first block
    let b = &gate_blocks[0];
    println!("First Q5_K block:");
    println!("  d = {:.6}", fp16_to_fp32(b.d));
    println!("  dmin = {:.6}", fp16_to_fp32(b.dmin));
    println!("  scales [0..5]: {:?}", &b.scales[..5]);
    println!("  qs [0..5]: {:?}", &b.qs[..5]);
    println!("  qh [0..5]: {:?}", &b.qh[..5]);
    
    // Create a test input
    let input: Vec<f32> = vec![1.0; QK_K];  // Simple input
    let mut input_q8k: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; 1];
    quantize_q8_k(&input, &mut input_q8k);
    
    println!("\nInput Q8_K block:");
    println!("  d = {:.6}", input_q8k[0].d);
    
    // Compute dot product
    let result = vec_dot_q5_k_q8_k(&gate_blocks[..1], &input_q8k);
    println!("\nResult: {:.6}", result);
    
    // Check scales distribution
    let mut scale_values: Vec<u8> = Vec::new();
    for b in &gate_blocks[..100] {
        scale_values.extend_from_slice(&b.scales);
    }
    let scale_min = scale_values.iter().cloned().min().unwrap();
    let scale_max = scale_values.iter().cloned().max().unwrap();
    println!("\nScales in first 100 blocks: min={}, max={}", scale_min, scale_max);
}
