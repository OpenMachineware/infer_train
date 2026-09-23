use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;
use std::time::Instant;

const QK_K: usize = 256;

fn generate_q8_k(k: usize) -> Vec<BlockQ8K> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut qs = [0i8; QK_K];
        for i in 0..QK_K {
            qs[i] = ((i + b * QK_K) % 256) as i8;
        }
        let mut bsums = [0i16; 16];
        for i in 0..16 {
            let start = i * 16;
            let mut sum: i32 = 0;
            for j in 0..16 {
                sum += qs[start + j] as i32;
            }
            bsums[i] = sum as i16;
        }
        blocks.push(BlockQ8K {
            d: 1.0,
            qs,
            bsums,
        });
    }
    blocks
}

fn generate_q2_k(k: usize) -> Vec<BlockQ2K> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut scales = [0u8; 16];
        for i in 0..16 {
            scales[i] = ((i + b) % 64) as u8;
        }
        let qs = [((b % 256) as u8); 64];
        blocks.push(BlockQ2K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.5).to_bits(),
            scales,
            qs,
        });
    }
    blocks
}

fn generate_q3_k(k: usize) -> Vec<BlockQ3K> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut hmask = [0u8; 32];
        for i in 0..32 {
            hmask[i] = ((i + b) % 256) as u8;
        }
        let qs = [((b % 256) as u8); 64];
        let mut scales = [0u8; 12];
        for i in 0..12 {
            scales[i] = ((i + b) % 64) as u8;
        }
        blocks.push(BlockQ3K {
            d: half::f16::from_f32(1.0).to_bits(),
            hmask,
            qs,
            scales,
        });
    }
    blocks
}

fn generate_q4_k(k: usize) -> Vec<BlockQ4K> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut scales = [0u8; 12];
        for i in 0..12 {
            scales[i] = ((i + b) % 64) as u8;
        }
        let qs = [((b % 256) as u8); 128];
        blocks.push(BlockQ4K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.5).to_bits(),
            scales,
            qs,
        });
    }
    blocks
}

fn generate_q5_k(k: usize) -> Vec<BlockQ5K> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut scales = [0u8; 12];
        for i in 0..12 {
            scales[i] = ((i + b) % 64) as u8;
        }
        let qh = [((b % 256) as u8); 32];
        let qs = [((b % 256) as u8); 128];
        blocks.push(BlockQ5K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.5).to_bits(),
            scales,
            qh,
            qs,
        });
    }
    blocks
}

fn generate_q6_k(k: usize) -> Vec<BlockQ6K> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for i in 0..128 {
            ql[i] = ((i + b) % 256) as u8;
        }
        for i in 0..64 {
            qh[i] = ((i + b + 128) % 256) as u8;
        }
        let mut scales = [0i8; 16];
        for i in 0..16 {
            scales[i] = ((i + b) % 128) as i8;
        }
        blocks.push(BlockQ6K {
            d: half::f16::from_f32(1.0).to_bits(),
            ql,
            qh,
            scales,
        });
    }
    blocks
}

fn generate_iq4_xs(k: usize) -> Vec<BlockIQ4XS> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let qs = [((b % 256) as u8); 128];
        let scales_l = [((b % 256) as u8); 4];
        blocks.push(BlockIQ4XS {
            d: half::f16::from_f32(1.0).to_bits(),
            scales_h: 0,
            scales_l,
            qs,
        });
    }
    blocks
}

fn generate_iq3_xxs(k: usize) -> Vec<BlockIQ3XXS> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let qs = [((b % 256) as u8); 96];
        blocks.push(BlockIQ3XXS {
            d: half::f16::from_f32(1.0).to_bits(),
            qs,
        });
    }
    blocks
}

fn generate_iq2_xxs(k: usize) -> Vec<BlockIQ2XXS> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let qs = [((b % 256) as u16); 32];
        blocks.push(BlockIQ2XXS {
            d: half::f16::from_f32(1.0).to_bits(),
            qs,
        });
    }
    blocks
}

fn generate_tq2_0(k: usize) -> Vec<BlockTQ2_0> {
    let nb = k / QK_K;
    let mut blocks = Vec::with_capacity(nb);
    for b in 0..nb {
        let qs = [((b % 256) as u8); 64];
        blocks.push(BlockTQ2_0 {
            d: half::f16::from_f32(1.0).to_bits(),
            qs,
        });
    }
    blocks
}

fn main() {
    println!("Multi-size vec_dot performance benchmark");
    println!("========================================");
    println!();

    let k_values = [256, 512, 1024, 2048, 4096];
    let iterations = 10000;

    // K-quant formats
    println!("K-Quant Formats:");
    println!("─────────────────────────────────────────────────────────────────────────────────────");
    println!("│ Format  │ K=256  │ K=512  │ K=1024 │ K=2048 │ K=4096 │ Avg    │ MaxDev │");
    println!("│         │ GFLOPS │ GFLOPS │ GFLOPS │ GFLOPS │ GFLOPS │ GFLOPS │   %    │");
    println!("─────────────────────────────────────────────────────────────────────────────────────");

    // Q2_K
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_q2_k(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_q2_k_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ Q2_K    │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    // Q3_K
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_q3_k(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_q3_k_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ Q3_K    │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    // Q4_K
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_q4_k(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_q4_k_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ Q4_K    │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    // Q5_K
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_q5_k(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_q5_k_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ Q5_K    │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    // Q6_K
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_q6_k(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_q6_k_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ Q6_K    │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    println!("─────────────────────────────────────────────────────────────────────────────────────");
    println!();

    // IQ/TQ formats
    println!("IQ/TQ Formats:");
    println!("─────────────────────────────────────────────────────────────────────────────────────");

    // IQ4_XS
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_iq4_xs(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_iq4_xs_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ IQ4_XS  │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    // IQ3_XXS
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_iq3_xxs(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_iq3_xxs_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ IQ3_XXS │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    // IQ2_XXS
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_iq2_xxs(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_iq2_xxs_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ IQ2_XXS │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    // TQ2_0
    {
        let mut avg_gflops = 0.0;
        let mut gflops_values = Vec::new();
        for &k in &k_values {
            let x = generate_q8_k(k);
            let y = generate_tq2_0(k);
            let t0 = Instant::now();
            for _ in 0..iterations {
                unsafe {
                    let _ = vec_dot_tq2_0_q8_k_neon(k, &y, &x);
                }
            }
            let elapsed = t0.elapsed().as_secs_f64();
            let gflops = (2.0 * k as f64 * iterations as f64) / elapsed / 1e9;
            gflops_values.push(gflops);
            avg_gflops += gflops;
        }
        avg_gflops /= k_values.len() as f64;
        let max_dev = gflops_values.iter()
            .map(|&g| ((g - avg_gflops).abs() / avg_gflops) * 100.0)
            .fold(0.0, f64::max);
        println!("│ TQ2_0   │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:6.1} │ {:5.1} │ {:4.1}% │",
            gflops_values[0], gflops_values[1], gflops_values[2], gflops_values[3], gflops_values[4], avg_gflops, max_dev);
    }

    println!("─────────────────────────────────────────────────────────────────────────────────────");
}
