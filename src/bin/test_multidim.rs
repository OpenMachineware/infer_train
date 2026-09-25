use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn main() {
    let ctx = MetalContext::new().unwrap();
    
    // 测试不同维度
    for (m, n) in [(1,1), (1,2), (2,1), (2,2), (4,4)].iter() {
        let k = 256;
        
        let weights: Vec<BlockQ4K> = (0..*m).map(|_| BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales: [1u8; 12],
            qs: [0b00001111u8; 128],
        }).collect();
        
        let input = vec![1.0f32; k * n];
        
        let result_hw = ctx.gemm_q4_k_f32(*m, *n, k, &weights, &input).unwrap();
        let result_tp = ctx.gemm_template_q4_k_f32(*m, *n, k, &weights, &input).unwrap();
        
        let expected = 1920.0;  // 单行单列的预期值
        
        print!("M={}, N={}: ", m, n);
        let hw_ok = result_hw.iter().all(|&x| (x - expected).abs() < 1.0);
        let tp_ok = result_tp.iter().all(|&x| (x - expected).abs() < 1.0);
        println!("HW={} TP={}", if hw_ok {"✓"} else {"✗"}, if tp_ok {"✓"} else {"✗"});
        
        if !tp_ok {
            println!("  TP values: {:?}", &result_tp[..4.min(result_tp.len())]);
        }
    }
}
