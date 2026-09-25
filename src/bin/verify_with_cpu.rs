// 用CPU计算作为ground truth
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn dequantize_q4_k_cpu(block: &BlockQ4K) -> Vec<f32> {
    // 简化的CPU dequantize
    let d = f16::from_bits(block.d).to_f32();
    let dmin = f16::from_bits(block.dmin).to_f32();
    
    let mut result = vec![0.0f32; 256];
    
    // Q4_K splits 256 values into 8 groups of 32
    // Each group has its own scale and min
    // For simplicity, assume all scales = 1, all mins = 0
    for i in 0..128 {
        let low = (block.qs[i] & 0xF) as f32;
        let high = (block.qs[i] >> 4) as f32;
        result[2*i] = d * 1.0 * low - dmin * 0.0;
        result[2*i + 1] = d * 1.0 * high - dmin * 0.0;
    }
    
    result
}

fn main() {
    let ctx = MetalContext::new().unwrap();
    
    let m = 2;
    let k = 256;
    let n = 2;
    
    // 简单的测试数据：d=1, dmin=0, qs=[15,15,...]
    let weights = vec![BlockQ4K {
        d: f16::from_f32(1.0).to_bits(),
        dmin: f16::from_f32(0.0).to_bits(),
        scales: [1u8; 12],  // Simplified
        qs: [15u8; 128],
    }, BlockQ4K {
        d: f16::from_f32(1.0).to_bits(),
        dmin: f16::from_f32(0.0).to_bits(),
        scales: [2u8; 12],  // Different scale
        qs: [15u8; 128],
    }];
    
    let input: Vec<f32> = vec![1.0; 512];  // k*n = 256*2
    
    // CPU计算
    let dequant0 = dequantize_q4_k_cpu(&weights[0]);
    let dequant1 = dequantize_q4_k_cpu(&weights[1]);
    
    println!("CPU dequant sample (first 10): {:?}", &dequant0[..10]);
    
    // CPU matmul: [2, 256] x [256, 2] = [2, 2]
    let mut cpu_result = vec![0.0f32; m * n];
    for i in 0..m {
        let dequant = if i == 0 { &dequant0 } else { &dequant1 };
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += dequant[l] * input[j * k + l];
            }
            cpu_result[i * n + j] = sum;
        }
    }
    println!("CPU result: {:?}", cpu_result);
    
    // GPU结果
    let result_hw = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
    println!("GPU hand-written: {:?}", result_hw);
    
    let result_tp = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).unwrap();
    println!("GPU templated: {:?}", result_tp);
    
    // 检查误差
    for i in 0..(m*n) {
        println!("Result[{}]: CPU={:.2}, HW={:.2}, TP={:.2}", 
                 i, cpu_result[i], result_hw[i], result_tp[i]);
    }
}
