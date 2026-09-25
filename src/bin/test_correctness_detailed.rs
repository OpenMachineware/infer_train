// 详细验证正确性
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn main() {
    let ctx = MetalContext::new().unwrap();
    
    // Test 1: 最小case，手动构造正确格式的block
    println!("=== Test 1: Minimal case with correct format ===");
    let m = 1;
    let k = 256;
    let n = 1;
    
    // Q4_K格式：
    // - d, dmin: FP16 scales
    // - scales[12]: packed scales and mins
    // - qs[128]: 4-bit quantized values (2 per byte)
    
    // 简化：设置所有scale为1，min为0
    let mut scales = [0u8; 12];
    for i in 0..6 {
        scales[i] = 1;     // scale
        scales[i+6] = 0;   // min
    }
    
    let weights = vec![BlockQ4K {
        d: f16::from_f32(1.0).to_bits(),
        dmin: f16::from_f32(0.0).to_bits(),
        scales,
        qs: [15u8; 128],  // 每个4-bit值=15
    }];
    
    let input = vec![1.0f32; k * n];
    
    let result_hw = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
    println!("Hand-written: {:?}", result_hw);
    
    let result_tp = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).unwrap();
    println!("Templated: {:?}", result_tp);
    
    // 预期：d=1, scale=1, q=15, min=0
    // dequant = d * scale * q - dmin * min = 1*1*15 - 0 = 15
    // sum over 256 values = 15 * 256 = 3840
    // dot with input [1.0; 256] = 3840
    println!("Expected: 3840 (1*1*15*256)");
    println!();
    
    // Test 2: 检查为什么Test 2的第一行是0
    println!("=== Test 2: Check why first row is zero ===");
    let m = 2;
    let k = 256;
    let n = 2;
    
    let weights: Vec<BlockQ4K> = (0..m).map(|i| {
        let mut scales = [0u8; 12];
        for j in 0..6 {
            scales[j] = ((i+1) * 2) as u8;  // scale varies by row
            scales[j+6] = 0;
        }
        BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs: [((i+1) * 10) as u8; 128],
        }
    }).collect();
    
    let input: Vec<f32> = (0..(k * n)).map(|i| (i % 10) as f32 * 0.1).collect();
    
    let result_hw = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
    let result_tp = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).unwrap();
    
    println!("Hand-written:");
    for i in 0..m {
        for j in 0..n {
            print!("{:.2} ", result_hw[i * n + j]);
        }
        println!();
    }
    
    println!("\nTemplated:");
    for i in 0..m {
        for j in 0..n {
            print!("{:.2} ", result_tp[i * n + j]);
        }
        println!();
    }
}
