// 分析线程到输出位置的映射
fn main() {
    let nl0 = 2;  // NK/16 = 32/16
    let nl1 = 4;  // NK/8 = 32/8
    let nr0 = 64; // 期望的行数
    let nr1 = 32; // 期望的列数

    println!("Thread mapping (tiitg -> (lr0, lr1, il0)):");
    println!("tiitg | lr0 = tiitg/{} | lr1 = tiitg/{} | il0 = tiitg%{}", nl0, nl1, nl0);
    println!("------|----------------|----------------|-------------");

    for tiitg in [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 127].iter() {
        let lr0 = tiitg / nl0;
        let lr1 = tiitg / nl1;
        let il0 = tiitg % nl0;
        println!("{:5} | {:14} | {:14} | {:11}", tiitg, lr0, lr1, il0);
    }

    println!("\n问题分析:");
    println!("  - lr0范围: 0..{} (需要{}行)", 127/nl0, nr0);
    println!("  - lr1范围: 0..{} (需要{}列)", 127/nl1, nr1);

    println!("\n关键问题:");
    println!("  lr0和lr1同时从tiitg计算，覆盖的是对角线！");
    println!("  tiitg=0: (lr0=0, lr1=0) -> 位置(0,0)");
    println!("  tiitg=1: (lr0=0, lr1=0) -> 位置(0,0) [重复]");
    println!("  tiitg=2: (lr0=1, lr1=0) -> 位置(1,0)");
    println!("  tiitg=4: (lr0=2, lr1=1) -> 位置(2,1)");

    println!("\n正确理解：");
    println!("  每个线程不是独立处理一个输出位置！");
    println!("  simdgroup内的32个线程协作计算8x8=64个值");
    println!("  lr0/lr1是加载weights/input的位置，不是输出位置");

    println!("\n实际流程:");
    println!("  1. 每个线程加载weights的一小部分（通过lr0）");
    println!("  2. 每个线程加载input的一小部分（通过lr1）");
    println!("  3. simdgroup协作计算矩阵乘法");
    println!("  4. simdgroup_store写入8x8结果");

    println!("\n所以问题在于：");
    println!("  - 每个threadgroup有4个simdgroups (sgitg=0,1,2,3)");
    println!("  - 每个simdgroup写入8x8x8=512个值？");
    println!("  - 但mc[8]是per-threadgroup还是per-simdgroup？");
}
