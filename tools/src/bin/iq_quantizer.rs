// IQ Quantizer Tool - Generate valid quantized test data
// Based on llama.cpp test-quantize-fns.cpp
//
// Usage:
//   cargo run --release --bin iq_quantizer -- --format iq2_xxs --output test.bin
//   cargo run --release --bin iq_quantizer -- --format iq3_s --blocks 10 --export-rust

use clap::Parser;
use half::f16;
use std::fs::File;
use std::io::Write;

#[derive(Parser)]
#[command(name = "iq_quantizer")]
#[command(about = "Generate quantized test data for IQ formats")]
struct Args {
    /// Format: iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs
    #[arg(short, long)]
    format: String,

    /// Number of blocks (default: 1)
    #[arg(short, long, default_value = "1")]
    blocks: usize,

    /// Output binary file
    #[arg(short, long)]
    output: Option<String>,

    /// Export as Rust source file
    #[arg(long)]
    export_rust: Option<String>,

    /// Random seed (default: 42)
    #[arg(short, long, default_value = "42")]
    seed: u64,
}

const QK_K: usize = 256; // Super-block size

// Generate synthetic data (matching llama.cpp test)
fn generate_data(offset: f32, n: usize, amplitude: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.1 + amplitude * ((i as f32) + offset).cos())
        .collect()
}

// ============================================================================
// Block structures (must match Metal/GPU kernel expectations)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ1S {
    d: u16,       // FP16 scale
    qs: [u8; 32], // Grid indices
    qh: [u16; 8], // High bits
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ1M {
    qs: [u8; 32],    // Grid indices, low 8 bits
    qh: [u8; 16],    // Grid indices, high bits
    scales: [u8; 8], // Scales
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ2XXS {
    d: u16,        // FP16 scale
    qs: [u16; 32], // Grid indices + signs
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ2XS {
    d: u16,          // FP16 scale
    qs: [u16; 32],   // Grid indices + signs
    scales: [u8; 8], // Scales
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ2S {
    d: u16,          // FP16 scale
    qs: [u8; 64],    // Grid indices
    qh: [u8; 8],     // High bits
    scales: [u8; 8], // Scales
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ3XXS {
    d: u16,      // FP16 scale
    qs: [u8; 96], // Grid indices packed
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ3S {
    d: u16,          // FP16 scale
    qs: [u8; 64],    // Grid indices
    qh: [u8; 8],     // High bits
    signs: [u8; 32], // Sign bits
    scales: [u8; 4], // Scales
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ4NL {
    d: u16,       // FP16 scale
    qs: [u8; 16], // 4-bit quants (32 values)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockIQ4XS {
    d: u16,          // FP16 scale
    scales_h: u16,   // High bits of scales
    scales_l: [u8; 4], // Low bits of scales
    qs: [u8; 128],   // 4-bit quants (256 values)
}

// ============================================================================
// Simplified quantization (for test data generation)
// These produce valid blocks, though not optimal quantization
// ============================================================================

fn quantize_iq4_nl(x: &[f32]) -> BlockIQ4NL {
    assert!(x.len() == 32);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = amax / 8.0;
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 16];
    for i in 0..16 {
        let x0 = x[i] * id;
        let x1 = x[i + 16] * id;
        let q0 = ((x0 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
        let q1 = ((x1 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
        qs[i] = q0 | (q1 << 4);
    }

    BlockIQ4NL {
        d: f16::from_f32(d).to_bits(),
        qs,
    }
}

fn quantize_iq4_xs(x: &[f32]) -> BlockIQ4XS {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = amax / 8.0;
    let id = if d > 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0u8; 128];
    for i in 0..128 {
        let x0 = x[i] * id;
        let x1 = x[i + 128] * id;
        let q0 = ((x0 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
        let q1 = ((x1 + 8.5).clamp(0.0, 15.0) as u8) & 0x0F;
        qs[i] = q0 | (q1 << 4);
    }

    BlockIQ4XS {
        d: f16::from_f32(d).to_bits(),
        scales_h: 0,
        scales_l: [0u8; 4],
        qs,
    }
}

fn quantize_iq2_xxs(x: &[f32]) -> BlockIQ2XXS {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = if amax > 0.0 { amax } else { 1.0 };

    // 2-bit quantization per value (simplified)
    let mut qs = [0u16; 32];
    for i in 0..32 {
        let mut idx = 0u16;
        for j in 0..8 {
            let val = x[i * 8 + j];
            let q = if val > d * 0.5 {
                3u16
            } else if val > 0.0 {
                2u16
            } else if val > -d * 0.5 {
                1u16
            } else {
                0u16
            };
            idx |= q << (2 * j);
        }
        qs[i] = idx;
    }

    BlockIQ2XXS {
        d: f16::from_f32(d).to_bits(),
        qs,
    }
}

fn quantize_iq2_xs(x: &[f32]) -> BlockIQ2XS {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = if amax > 0.0 { amax } else { 1.0 };

    let mut qs = [0u16; 32];
    let scales = [0u8; 8]; // Simplified

    for i in 0..32 {
        let mut idx = 0u16;
        for j in 0..8 {
            let val = x[i * 8 + j];
            let q = if val > d * 0.5 {
                3u16
            } else if val > 0.0 {
                2u16
            } else if val > -d * 0.5 {
                1u16
            } else {
                0u16
            };
            idx |= q << (2 * j);
        }
        qs[i] = idx;
    }

    BlockIQ2XS {
        d: f16::from_f32(d).to_bits(),
        qs,
        scales,
    }
}

fn quantize_iq2_s(x: &[f32]) -> BlockIQ2S {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = if amax > 0.0 { amax } else { 1.0 };

    let mut qs = [0u8; 64];
    let qh = [0u8; 8];
    let scales = [0u8; 8];

    for i in 0..64 {
        let mut byte = 0u8;
        for j in 0..4 {
            let val = x[i * 4 + j];
            let q = if val > d * 0.5 {
                3u8
            } else if val > 0.0 {
                2u8
            } else if val > -d * 0.5 {
                1u8
            } else {
                0u8
            };
            byte |= q << (2 * j);
        }
        qs[i] = byte;
    }

    BlockIQ2S {
        d: f16::from_f32(d).to_bits(),
        qs,
        qh,
        scales,
    }
}

fn quantize_iq3_xxs(x: &[f32]) -> BlockIQ3XXS {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = if amax > 0.0 { amax } else { 1.0 };

    // Simplified 3-bit packing
    let mut qs = [0u8; 96];
    for i in 0..96 {
        let idx = (i * 8) / 3;
        if idx < 256 {
            let val = x[idx];
            let q = ((val / d + 1.0) * 3.5).clamp(0.0, 7.0) as u8;
            qs[i] = q;
        }
    }

    BlockIQ3XXS {
        d: f16::from_f32(d).to_bits(),
        qs,
    }
}

fn quantize_iq3_s(x: &[f32]) -> BlockIQ3S {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = if amax > 0.0 { amax } else { 1.0 };

    let mut qs = [0u8; 64];
    let qh = [0u8; 8];
    let signs = [0u8; 32];
    let scales = [0u8; 4];

    for i in 0..64 {
        let mut byte = 0u8;
        for j in 0..4 {
            let val = x[i * 4 + j];
            let q = ((val / d + 1.0) * 3.5).clamp(0.0, 7.0) as u8;
            byte |= q << (3 * (j % 2));
        }
        qs[i] = byte;
    }

    BlockIQ3S {
        d: f16::from_f32(d).to_bits(),
        qs,
        qh,
        signs,
        scales,
    }
}

fn quantize_iq1_s(x: &[f32]) -> BlockIQ1S {
    assert!(x.len() == 256);

    let mut amax = 0.0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }

    let d = if amax > 0.0 { amax } else { 1.0 };

    // 1-bit sign only (simplified)
    let mut qs = [0u8; 32];
    let qh = [0u16; 8];

    for i in 0..32 {
        let mut byte = 0u8;
        for j in 0..8 {
            if x[i * 8 + j] > 0.0 {
                byte |= 1 << j;
            }
        }
        qs[i] = byte;
    }

    BlockIQ1S {
        d: f16::from_f32(d).to_bits(),
        qs,
        qh,
    }
}

fn quantize_iq1_m(x: &[f32]) -> BlockIQ1M {
    assert!(x.len() == 256);

    let mut qs = [0u8; 32];
    let qh = [0u8; 16];
    let scales = [0u8; 8];

    for i in 0..32 {
        let mut byte = 0u8;
        for j in 0..8 {
            if x[i * 8 + j] > 0.0 {
                byte |= 1 << j;
            }
        }
        qs[i] = byte;
    }

    BlockIQ1M { qs, qh, scales }
}

// ============================================================================
// Main logic
// ============================================================================

fn main() {
    let args = Args::parse();

    let format = args.format.to_lowercase();
    let n_blocks = args.blocks;
    let seed = args.seed;

    // Determine block size
    let block_size = if format == "iq4_nl" { 32 } else { 256 };

    println!("=== IQ Quantizer ===");
    println!("Format: {}", format);
    println!("Blocks: {}", n_blocks);
    println!("Block size: {}", block_size);
    println!("Seed: {}", seed);
    println!();

    // Generate test data
    let total_values = n_blocks * block_size;
    let fp32_data = generate_data(seed as f32, total_values, 2.0);

    // Quantize blocks
    let mut quantized_bytes = Vec::new();

    for block_idx in 0..n_blocks {
        let start = block_idx * block_size;
        let block_data = &fp32_data[start..start + block_size];

        let block_bytes = match format.as_str() {
            "iq1_s" => {
                let block = quantize_iq1_s(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ1S>())
                }.to_vec()
            }
            "iq1_m" => {
                let block = quantize_iq1_m(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ1M>())
                }.to_vec()
            }
            "iq2_xxs" => {
                let block = quantize_iq2_xxs(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ2XXS>())
                }.to_vec()
            }
            "iq2_xs" => {
                let block = quantize_iq2_xs(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ2XS>())
                }.to_vec()
            }
            "iq2_s" => {
                let block = quantize_iq2_s(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ2S>())
                }.to_vec()
            }
            "iq3_xxs" => {
                let block = quantize_iq3_xxs(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ3XXS>())
                }.to_vec()
            }
            "iq3_s" => {
                let block = quantize_iq3_s(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ3S>())
                }.to_vec()
            }
            "iq4_nl" => {
                let block = quantize_iq4_nl(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ4NL>())
                }.to_vec()
            }
            "iq4_xs" => {
                let block = quantize_iq4_xs(block_data);
                unsafe {
                    std::slice::from_raw_parts(&block as *const _ as *const u8, std::mem::size_of::<BlockIQ4XS>())
                }.to_vec()
            }
            _ => {
                eprintln!("Unknown format: {}", format);
                std::process::exit(1);
            }
        };

        quantized_bytes.extend(block_bytes);
    }

    println!("Generated {} bytes", quantized_bytes.len());

    // Save binary output
    if let Some(output_path) = args.output {
        let mut file = File::create(&output_path).expect("Failed to create output file");
        file.write_all(&quantized_bytes).expect("Failed to write output");
        println!("Saved to {}", output_path);
    }

    // Export as Rust source
    if let Some(rust_path) = args.export_rust {
        let mut file = File::create(&rust_path).expect("Failed to create Rust file");

        writeln!(file, "// Auto-generated test data for {}", format).unwrap();
        writeln!(file, "// Seed: {}, Blocks: {}", seed, n_blocks).unwrap();
        writeln!(file).unwrap();
        writeln!(file, "pub const FORMAT: &str = \"{}\";", format).unwrap();
        writeln!(file, "pub const BLOCK_SIZE: usize = {};", block_size).unwrap();
        writeln!(file, "pub const N_BLOCKS: usize = {};", n_blocks).unwrap();
        writeln!(file).unwrap();

        writeln!(file, "pub const FP32_DATA: &[f32] = &[").unwrap();
        for (i, v) in fp32_data.iter().enumerate() {
            if i % 8 == 0 {
                write!(file, "    ").unwrap();
            }
            write!(file, "{:.6}, ", v).unwrap();
            if (i + 1) % 8 == 0 {
                writeln!(file).unwrap();
            }
        }
        writeln!(file, "];").unwrap();
        writeln!(file).unwrap();

        writeln!(file, "pub const QUANTIZED_DATA: &[u8] = &[").unwrap();
        for (i, b) in quantized_bytes.iter().enumerate() {
            if i % 16 == 0 {
                write!(file, "    ").unwrap();
            }
            write!(file, "0x{:02x}, ", b).unwrap();
            if (i + 1) % 16 == 0 {
                writeln!(file).unwrap();
            }
        }
        writeln!(file, "];").unwrap();

        println!("Exported to {}", rust_path);
    }

    println!("\nDone.");
}
