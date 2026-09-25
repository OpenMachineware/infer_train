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
    
    // 打印前几行的非零列位置
    for row in [0, 31, 32, 63, 64, 95, 96, 127].iter() {
        let row = *row;
        let row_start = row * n;
        let row_end = row_start + n;
        let first_nonzero = result[row_start..row_end].iter().position(|&x| x != 0.0);
        let last_nonzero = result[row_start..row_end].iter().rposition(|&x| x != 0.0);
        println!("Row {:3}: cols {:?} .. {:?}", row, first_nonzero, last_nonzero);
    }
    
    // 计算总数
    let total_nonzero = result.iter().filter(|&&x| x != 0.0).count();
    println!("\nTotal non-zero: {}/{} = {:.1}%", total_nonzero, m*n, 100.0 * total_nonzero as f64 / (m*n) as f64);
    
    // 这4096个值来自哪些threadgroup?
    // Grid: 4x2 (width=N方向, height=M方向)
    // Threadgroup (y=0,x=0): rows 0-63, cols 0-31
    // Threadgroup (y=0,x=1): rows 0-63, cols 32-63
    // Threadgroup (y=0,x=2): rows 0-63, cols 64-95
    // Threadgroup (y=0,x=3): rows 0-63, cols 96-127
    // Threadgroup (y=1,x=0): rows 64-127, cols 0-31
    // ...
    
    // 实际情况：rows 0-31, 64-95 和 cols 0-63 有值
    // 这对应：2个M方向的threadgroup × 2个N方向的threadgroup × 32×64个值 = 4096
    println!("\nActual pattern: rows 0-31,64-95 x cols 0-63");
    println!("Expected: 64 rows × 64 cols = 4096 ✓");
}
