// 分析kernel的线程分配逻辑
use metal::{Device, MTLSize};

fn main() {
    // 当前配置
    let m = 128;
    let n = 128;
    let nr0 = 64;  // kernel中定义
    let nr1 = 32;  // kernel中定义
    let threadgroup_size = 128;
    
    // Grid配置
    let grid_width = (n + 31) / 32;  // 4
    let grid_height = (m + 63) / 64; // 2
    
    println!("Current configuration:");
    println!("  Grid: {} × {} = {} threadgroups", grid_width, grid_height, grid_width * grid_height);
    println!("  Threadgroup size: {} threads", threadgroup_size);
    println!("  Expected output per threadgroup: {} rows × {} cols = {} values", nr0, nr1, nr0 * nr1);
    println!("  Total expected: {} values", grid_width * grid_height * nr0 * nr1);
    
    // Kernel中的线程分配
    // NK = 32 (K direction per iteration)
    // NL0 = NK/16 = 2
    // NL1 = NK/8 = 4
    
    // lr0 = tiitg / NL0 (row index within threadgroup)
    // lr1 = tiitg / NL1 (col index within threadgroup)
    // il0 = tiitg % NL0
    
    println!("\nKernel thread mapping:");
    println!("  tiitg range: 0..{}", threadgroup_size);
    println!("  lr0 = tiitg / 2 (max {} for tiitg=127)", 127/2);
    println!("  lr1 = tiitg / 4 (max {} for tiitg=127)", 127/4);
    
    // 问题：只有128个线程，但lr0最大=63（需要64个），lr1最大=31（需要32个）
    // 128 threads = 64 rows × 2 K iterations
    
    // 实际覆盖
    println!("\nActual coverage:");
    println!("  lr0: 0..63 (covers all {} rows)", nr0);
    println!("  lr1: 0..31 (covers all {} cols)", nr1);
    println!("  Wait, 128 threads can't cover 64×32 positions!");
    
    // 重新计算
    let max_lr0 = (threadgroup_size - 1) / 2;
    let max_lr1 = (threadgroup_size - 1) / 4;
    println!("\n  With {} threads:", threadgroup_size);
    println!("    lr0 max = {} (covers rows 0..{})", max_lr0, max_lr0);
    println!("    lr1 max = {} (covers cols 0..{})", max_lr1, max_lr1);
    
    // 实际线程分配
    // tiitg: 0, 1, 2, 3, ..., 127
    // lr0 = tiitg / 2: 0, 0, 1, 1, ..., 63, 63
    // lr1 = tiitg / 4: 0, 0, 0, 0, 1, 1, 1, 1, ..., 31, 31, 31, 31
    
    println!("\n  So each thread covers:");
    println!("    lr0: 64 positions (0..63)");
    println!("    lr1: 32 positions (0..31)");
    println!("    But only 128 threads -> each (lr0, lr1) pair appears multiple times");
    
    // 检查是否有足够的线程处理2048个输出位置
    // simdgroup存储：8个mc[i]，每个8×8=64个值
    // 8 × 64 = 512个值，但每个simdgroup有32个线程
    // 128 threads = 4 simdgroups
    // 4 simdgroups × 512 = 2048 values ✓
    
    println!("\n  Simdgroup analysis:");
    println!("    128 threads = 4 simdgroups (32 threads each)");
    println!("    Each simdgroup computes 8×8=64 output values");
    println!("    8 mc matrices × 64 values = 512 per threadgroup? No, 8 mc per SIMDGROUP");
    
    // 重新理解：每个THREADGROUP有8个simdgroups (256 threads)
    // 但我们只有128 threads = 4 simdgroups
    // 所以只能写入4 × 512 = 2048 values
    // 但kernel中只定义了8个mc，不是每个simdgroup一组？
}
