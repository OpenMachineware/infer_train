// 检查哪些位置有值
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn main() {
    let ctx = MetalContext::new().unwrap();
    
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
    
    let result = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).unwrap();
    
    // 打印哪些行有值
    println!("Non-zero pattern (M=128 rows, N=128 cols):");
    for row in 0..m {
        let row_start = row * n;
        let row_end = row_start + n;
        let nonzero_count = result[row_start..row_end].iter().filter(|&&x| x != 0.0).count();
        if nonzero_count > 0 {
            print!("Row {:3}: {} cols non-zero | ", row, nonzero_count);
            if row % 4 == 3 { println!(); }
        }
    }
    println!("\n");
    
    // 打印哪些列有值
    println!("Col non-zero pattern:");
    for col in 0..16 {
        let mut nonzero = 0;
        for row in 0..m {
            if result[row * n + col] != 0.0 {
                nonzero += 1;
            }
        }
        print!("Col {:3}: {} rows | ", col, nonzero);
        if col % 8 == 7 { println!(); }
    }
}
