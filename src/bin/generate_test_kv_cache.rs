// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Generate real quantized test data for flash decode benchmark

use infer_train::quant::{
    quantize_row_q4_0, quantize_row_q4_1, quantize_row_q5_0, quantize_row_q5_1,
    quantize_row_q8_0,
};

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;

fn main() {
    let seq_ks = [256, 512, 1024, 2048];

    for seq_k in &seq_ks {
        let num_elements = NUM_HEADS * seq_k * HEAD_DIM;

        // Generate random test data
        let mut data: Vec<f32> = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            // Generate values in range [-1, 1]
            data.push(((i % 200) as f32 / 100.0) - 1.0);
        }

        println!("Generating test data for seq_k={}", seq_k);

        // Quantize for each format
        let formats = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"];
        for format in &formats {
            let quantized = match *format {
                "q4_0" => {
                    let mut dst = vec![0u8; (num_elements / 32) * 18];
                    quantize_row_q4_0(&data, &mut dst);
                    dst
                }
                "q4_1" => {
                    let mut dst = vec![0u8; (num_elements / 32) * 20];
                    quantize_row_q4_1(&data, &mut dst);
                    dst
                }
                "q5_0" => {
                    let mut dst = vec![0u8; (num_elements / 32) * 22];
                    quantize_row_q5_0(&data, &mut dst);
                    dst
                }
                "q5_1" => {
                    let mut dst = vec![0u8; (num_elements / 32) * 24];
                    quantize_row_q5_1(&data, &mut dst);
                    dst
                }
                "q8_0" => {
                    let mut dst = vec![0u8; (num_elements / 32) * 34];
                    quantize_row_q8_0(&data, &mut dst);
                    dst
                }
                _ => panic!("Unknown format: {}", format),
            };

            // Save to file
            let filename = format!("test_data/{}_seq{}.bin", format, seq_k);
            std::fs::create_dir_all("test_data").ok();
            std::fs::write(&filename, &quantized)
                .expect("Failed to write file");
            println!(
                "  {} -> {} ({} bytes)",
                format,
                filename,
                quantized.len()
            );
        }

        // Save F16 data
        let f16_data: Vec<u16> =
            data.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let f16_bytes: Vec<u8> =
            f16_data.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let filename = format!("test_data/f16_seq{}.bin", seq_k);
        std::fs::write(&filename, &f16_bytes).expect("Failed to write file");
        println!("  f16 -> {} ({} bytes)", filename, f16_bytes.len());

        // Save FP32 query data
        let filename = format!("test_data/query_seq{}.bin", seq_k);
        let q_bytes: Vec<u8> = data[..NUM_HEADS * HEAD_DIM]
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        std::fs::write(&filename, &q_bytes).expect("Failed to write file");
        println!("  query -> {} ({} bytes)", filename, q_bytes.len());
    }

    println!("\nTest data generated in test_data/ directory");
}
