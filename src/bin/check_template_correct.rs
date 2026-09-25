// 验证templated kernel的正确性
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn main() {
    let ctx = MetalContext::new().unwrap();
    
    // 测试1：最小case - 正确设置input大小
    println!("=== Test 1: Minimal case (M=1, K=256, N=1) ===");
    println!("Input size should be K*N = 256*1 = 256");
    let m = 1;
    let k = 256;
    let n = 1;
    
    let weights = vec![BlockQ4K {
        d: f16::from_f32(1.0).to_bits(),
        dmin: f16::from_f32(0.0).to_bits(),
        scales: [1u8; 12],
        qs: [15u8; 128],  // 所有4-bit值都是15
    }];
    
    // 正确：K*N个输入
    let input = vec![1.0f32; k * n];
    
    let result_hw = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
    println!("Hand-written result: {:?}", result_hw);
    
    let result_tp = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).unwrap();
    println!("Templated result: {:?}", result_tp);
    println!("Expected: ~3840 (1*1*15*256)");
    
    // 测试2：检查输出覆盖
    println!("\n=== Test 2: Check output coverage (M=128, K=256, N=128) ===");
    let m = 128;
    let k = 256;
    let n = 128;
    
    let weights: Vec<BlockQ4K> = (0..m).map(|i| BlockQ4K {
        d: f16::from_f32(1.0).to_bits(),
        dmin: f16::from_f32(0.0).to_bits(),
        scales: [((i % 16) + 1) as u8; 12],
        qs: [((i * 17) % 256) as u8; 128],
    }).collect();
    
    let input: Vec<f32> = (0..(k * n)).map(|i| (i % 10) as f32).collect();
    
    let result_hw = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
    let result_tp = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).unwrap();
    
    // 统计非零值
    let hw_nonzero = result_hw.iter().filter(|&&x| x != 0.0).count();
    let tp_nonzero = result_tp.iter().filter(|&&x| x != 0.0).count();
    
    println!("Hand-written: {} non-zero out of {} values", hw_nonzero, result_hw.len());
    println!("Templated: {} non-zero out of {} values", tp_nonzero, result_tp.len());
    
    // 检查前几行
    println!("\nFirst row (should be same):");
    println!("HW: {:?}", &result_hw[..10.min(10)]);
    println!("TP: {:?}", &result_tp[..10.min(10)]);
    
    // 差异分析
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (a, b) in result_hw.iter().zip(result_tp.iter()) {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    println!("\nMax diff: {}", max_diff);
    println!("Avg diff: {}", sum_diff / result_hw.len() as f32);
}