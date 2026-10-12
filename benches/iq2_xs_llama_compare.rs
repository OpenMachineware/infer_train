// 直接调用 Rust 的 IQ2_XS 量化函数，与 llama.cpp dump 对比
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use std::fs;

fn main() {
    println!("=== Rust IQ2_XS vs llama.cpp 对比 ===\n");

    // 读取 llama.cpp dump 文件
    let dump_content = fs::read_to_string("/tmp/iq2_xs_llama_dump.txt")
        .expect("无法读取 dump 文件");

    // 解析 dump 文件
    let llama_values = parse_llama_dump(&dump_content);

    println!("llama.cpp values:");
    println!("  sigma2: {:.6}", llama_values.sigma2);
    println!("  weight[0:8]: {:?}", &llama_values.weight[0..8]);
    println!("  waux[0:8]: {:?}", &llama_values.waux[0..8]);
    println!("  xval[0:8]: {:?}", &llama_values.xval[0..8]);
    println!("  block_signs: {:?}", llama_values.block_signs);

    // 测试数据（与 llama.cpp 测试程序相同，扩展到 256）
    let mut test_data: Vec<f32> = vec![
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812,
        0.934245, 0.814190, -0.321567, 0.456789, -0.234567, 0.876543,
        -0.543210, 0.123456, -0.789012, 0.345678, 0.567890, -0.234567,
        0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678,
        0.567890, -0.234567,
    ];
    // 补齐到 256
    test_data.extend(vec![0.0f32; 256 - 32]);

    println!("\nRust 量化...");

    // 调用 Rust 量化函数（带 dump）
    use infer_train::quant::weights::iq::quantize_iq2_xs_with_dump;

    let quant_weights: Vec<f32> = vec![1.0; 256];
    let blocks = quantize_iq2_xs_with_dump(
        &test_data,
        Some(&quant_weights),
        Some("/tmp/iq2_xs_rust_dump.txt"),
    );

    println!("Rust quantization complete");
    println!("  Block d = 0x{:04x}", blocks[0].d);
    println!("  Block qs[0:8] = {:04x?}", &blocks[0].qs[0..8]);
    println!("  Block scales[0:8] = {:02x?}", &blocks[0].scales[0..8]);

    // 读取 Rust dump 并对比
    let rust_dump = fs::read_to_string("/tmp/iq2_xs_rust_dump.txt")
        .expect("无法读取 Rust dump");
    println!("\n=== Rust dump ===");
    println!("{}", rust_dump);

    // 对比
    println!("\n=== 对比结果 ===");
    compare_dumps(&llama_values, &rust_dump);
}

fn compare_dumps(llama: &LlamaDumpValues, rust_dump: &str) {
    let rust_values = parse_llama_dump(rust_dump);

    println!("sigma2:");
    println!("  llama: {:.6}", llama.sigma2);
    println!("  rust:  {:.6}", rust_values.sigma2);
    println!("  diff:  {:.9}", (llama.sigma2 - rust_values.sigma2).abs());

    println!("\nweight[0:8]:");
    for i in 0..8 {
        let diff = (llama.weight[i] - rust_values.weight[i]).abs();
        println!(
            "  [{}]: llama={:.6}, rust={:.6}, diff={:.9}",
            i, llama.weight[i], rust_values.weight[i], diff
        );
    }

    println!("\nwaux[0:8]:");
    for i in 0..8 {
        let diff = (llama.waux[i] - rust_values.waux[i]).abs();
        println!(
            "  [{}]: llama={:.6}, rust={:.6}, diff={:.9}",
            i, llama.waux[i], rust_values.waux[i], diff
        );
    }

    println!("\nxval[0:8]:");
    for i in 0..8 {
        let diff = (llama.xval[i] - rust_values.xval[i]).abs();
        println!(
            "  [{}]: llama={:.6}, rust={:.6}, diff={:.9}",
            i, llama.xval[i], rust_values.xval[i], diff
        );
    }

    println!("\nblock_signs:");
    println!("  llama: {:?}", llama.block_signs);
    println!("  rust:  {:?}", rust_values.block_signs);
    println!("  match: {}", llama.block_signs == rust_values.block_signs);
}

struct LlamaDumpValues {
    sigma2: f32,
    weight: [f32; 16],
    waux: [f32; 16],
    xval: [f32; 16],
    block_signs: [u8; 2],
}

fn parse_llama_dump(content: &str) -> LlamaDumpValues {
    let mut sigma2 = 0.0f32;
    let mut weight = [0.0f32; 16];
    let mut waux = [0.0f32; 16];
    let mut xval = [0.0f32; 16];
    let mut block_signs = [0u8; 2];

    for line in content.lines() {
        let line = line.trim();

        if line.contains("sigma2 = ") {
            // 格式: sumx2 = 9.887529, sigma2 = 0.038623
            let parts: Vec<&str> = line.split("sigma2 = ").collect();
            if parts.len() >= 2 {
                sigma2 = parts[1].trim().parse().unwrap_or(0.0);
            }
        } else if line.starts_with("weight[0:8]:") {
            let parts: Vec<&str> = line
                .trim_start_matches("weight[0:8]:")
                .split_whitespace()
                .collect();
            for (i, p) in parts.iter().enumerate().take(8) {
                weight[i] = p.parse().unwrap_or(0.0);
            }
        } else if line.starts_with("waux[0:8]:") {
            let parts: Vec<&str> = line
                .trim_start_matches("waux[0:8]:")
                .split_whitespace()
                .collect();
            for (i, p) in parts.iter().enumerate().take(8) {
                waux[i] = p.parse().unwrap_or(0.0);
            }
        } else if line.starts_with("xval[0:8]:") {
            let parts: Vec<&str> = line
                .trim_start_matches("xval[0:8]:")
                .split_whitespace()
                .collect();
            for (i, p) in parts.iter().enumerate().take(8) {
                xval[i] = p.parse().unwrap_or(0.0);
            }
        } else if line.starts_with("block_signs: ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            for (i, p) in parts[1..=2].iter().enumerate() {
                block_signs[i] = p.parse().unwrap_or(0);
            }
        }
    }

    LlamaDumpValues { sigma2, weight, waux, xval, block_signs }
}
