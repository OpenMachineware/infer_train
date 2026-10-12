// Test platform detection
use infer_train::platform;

fn main() {
    // Initialize platform detection
    platform::init();

    let cpu = platform::cpu();
    let gpu = platform::gpu();

    println!("=== Platform Detection ===\n");

    println!("CPU:");
    println!("  Architecture: {:?}", cpu.arch);
    println!("  Vendor: {}", cpu.vendor);
    println!("  Model: {}", cpu.model);
    println!("  Cores: {}", cpu.cores);
    println!("  L1 Data Cache: {} KB", cpu.l1_data_cache / 1024);
    println!("  L2 Cache: {} KB", cpu.l2_cache / 1024);

    if cpu.arch == platform::CpuArch::Arm {
        println!("  ARM Features:");
        println!("    NEON: {}", cpu.arm.neon);
        println!("    dotprod: {}", cpu.arm.dotprod);
        println!("    FP16: {}", cpu.arm.fp16);
        println!("    BF16: {}", cpu.arm.bf16);
    }

    if cpu.arch == platform::CpuArch::X86 {
        println!("  x86 Features:");
        println!("    AVX: {}", cpu.x86.avx);
        println!("    AVX2: {}", cpu.x86.avx2);
        println!("    AVX-512F: {}", cpu.x86.avx512f);
    }

    println!("\nGPU:");
    println!("  Vendor: {:?}", gpu.vendor);
    println!("  Name: {}", gpu.name);
    println!("  Memory: {} GB", gpu.memory / (1024 * 1024 * 1024));
    println!("  SIMD Width: {}", gpu.simd_width);
    println!(
        "  Max Threadgroup Memory: {} KB",
        gpu.max_threadgroup_memory / 1024
    );
    println!("  Max SIMD Groups: {}", gpu.max_simd_groups);
    println!("  Features:");
    println!("    SIMD dot: {}", gpu.has_simd_dot);
    println!("    BF16: {}", gpu.has_bfloat);
    println!("    FP16: {}", gpu.has_fp16);
    println!("    Int8 matmul: {}", gpu.has_int8_matmul);

    println!("\nSummary: {} | {}", cpu, gpu);
}
