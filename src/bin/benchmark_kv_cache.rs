// Benchmark KV cache quantization vs llama.cpp
use infer_train::quant::kv_cache::*;
use std::time::Instant;

fn main() {
    println!("=== KV Cache Quantization Benchmark vs llama.cpp ===\n");

    let sizes = [1024, 4096, 16384, 65536];
    let runs = 10;

    println!("Format | Size  | Our Quant | llama Quant | Our Dequant | llama Dequant | Ratio");
    println!("-------+-------+-----------+-------------+-------------+---------------+-------");

    for &size in &sizes {
        // Q8_0
        benchmark_q8_0(size, runs);

        // Q4_0
        benchmark_q4_0(size, runs);

        // Q4_1
        benchmark_q4_1(size, runs);

        // Q5_0
        benchmark_q5_0(size, runs);

        // Q5_1
        benchmark_q5_1(size, runs);
    }
}


fn benchmark_q8_0(size: usize, runs: usize) {
    assert!(size % QK == 0);
    let nb = size / QK;

    let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut output_q8 = vec![BlockQ8_0 { d: 0, qs: [0i8; QK] }; nb];
    let mut output_f32 = vec![0.0f32; size];

    // Our quantize
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q8_0(&input, &mut output_q8);
    }
    let our_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_quant_gbps = (size * 4) as f64 / our_quant_time / 1e9;

    // llama.cpp quantize (scalar reference)
    let mut output_q8_llama = vec![BlockQ8_0 { d: 0, qs: [0i8; QK] }; nb];
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q8_0_llama(&input, &mut output_q8_llama);
    }
    let llama_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_quant_gbps = (size * 4) as f64 / llama_quant_time / 1e9;

    // Verify quantization correctness
    let quant_match = output_q8.iter().zip(output_q8_llama.iter()).all(|(a, b)| {
        a.d == b.d && a.qs == b.qs
    });

    // Our dequantize
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q8_0(&output_q8, &mut output_f32);
    }
    let our_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_dequant_gbps = (size * 4) as f64 / our_dequant_time / 1e9;

    // llama.cpp dequantize (scalar reference)
    let mut output_f32_llama = vec![0.0f32; size];
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q8_0_llama(&output_q8_llama, &mut output_f32_llama);
    }
    let llama_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_dequant_gbps = (size * 4) as f64 / llama_dequant_time / 1e9;

    // Verify dequantization correctness
    let dequant_match = output_f32.iter().zip(output_f32_llama.iter()).all(|(a, b)| {
        (a - b).abs() < 1e-5
    });

    let quant_ratio = our_quant_gbps / llama_quant_gbps * 100.0;
    let dequant_ratio = our_dequant_gbps / llama_dequant_gbps * 100.0;

    println!("Q8_0   | {:5} | {:7.1} GB | {:9.1} GB | {:9.1} GB | {:11.1} GB | {:4.0}%/{:4.0}% | Q:{} D:{}",
        size, our_quant_gbps, llama_quant_gbps, our_dequant_gbps, llama_dequant_gbps,
        quant_ratio, dequant_ratio, quant_match, dequant_match);
}

fn benchmark_q4_0(size: usize, runs: usize) {
    assert!(size % QK == 0);
    let nb = size / QK;

    let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut output_q4 = vec![BlockQ4_0 { d: 0, qs: [0u8; QK / 2] }; nb];
    let mut output_f32 = vec![0.0f32; size];

    // Our quantize
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q4_0(&input, &mut output_q4);
    }
    let our_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_quant_gbps = (size * 4) as f64 / our_quant_time / 1e9;

    // llama.cpp quantize (scalar reference)
    let mut output_q4_llama = vec![BlockQ4_0 { d: 0, qs: [0u8; QK / 2] }; nb];
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q4_0_llama(&input, &mut output_q4_llama);
    }
    let llama_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_quant_gbps = (size * 4) as f64 / llama_quant_time / 1e9;

    // Our dequantize
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q4_0(&output_q4, &mut output_f32);
    }
    let our_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_dequant_gbps = (size * 4) as f64 / our_dequant_time / 1e9;

    // llama.cpp dequantize (scalar reference)
    let mut output_f32_llama = vec![0.0f32; size];
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q4_0_llama(&output_q4_llama, &mut output_f32_llama);
    }
    let llama_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_dequant_gbps = (size * 4) as f64 / llama_dequant_time / 1e9;

    let quant_match = output_q4.iter().zip(output_q4_llama.iter()).all(|(a, b)| {
        a.d == b.d && a.qs == b.qs
    });
    let dequant_match = output_f32.iter().zip(output_f32_llama.iter()).all(|(a, b)| {
        (a - b).abs() < 1e-5
    });

    let quant_ratio = our_quant_gbps / llama_quant_gbps * 100.0;
    let dequant_ratio = our_dequant_gbps / llama_dequant_gbps * 100.0;

    println!("Q4_0   | {:5} | {:7.1} GB | {:9.1} GB | {:9.1} GB | {:11.1} GB | {:4.0}%/{:4.0}% | Q:{} D:{}",
        size, our_quant_gbps, llama_quant_gbps, our_dequant_gbps, llama_dequant_gbps,
        quant_ratio, dequant_ratio, quant_match, dequant_match);
}

fn benchmark_q4_1(size: usize, runs: usize) {
    assert!(size % QK == 0);
    let nb = size / QK;

    let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut output_q4 = vec![BlockQ4_1 { d: 0, m: 0, qs: [0u8; QK / 2] }; nb];
    let mut output_f32 = vec![0.0f32; size];

    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q4_1(&input, &mut output_q4);
    }
    let our_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_quant_gbps = (size * 4) as f64 / our_quant_time / 1e9;

    let mut output_q4_llama = vec![BlockQ4_1 { d: 0, m: 0, qs: [0u8; QK / 2] }; nb];
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q4_1_llama(&input, &mut output_q4_llama);
    }
    let llama_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_quant_gbps = (size * 4) as f64 / llama_quant_time / 1e9;

    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q4_1(&output_q4, &mut output_f32);
    }
    let our_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_dequant_gbps = (size * 4) as f64 / our_dequant_time / 1e9;

    let mut output_f32_llama = vec![0.0f32; size];
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q4_1_llama(&output_q4_llama, &mut output_f32_llama);
    }
    let llama_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_dequant_gbps = (size * 4) as f64 / llama_dequant_time / 1e9;

    // Verify correctness
    let quant_match = output_q4.iter().zip(output_q4_llama.iter()).all(|(a, b)| {
        a.d == b.d && a.m == b.m && a.qs == b.qs
    });
    let dequant_match = output_f32.iter().zip(output_f32_llama.iter()).all(|(a, b)| {
        (a - b).abs() < 1e-5
    });

    let quant_ratio = our_quant_gbps / llama_quant_gbps * 100.0;
    let dequant_ratio = our_dequant_gbps / llama_dequant_gbps * 100.0;

    println!("Q4_1   | {:5} | {:7.1} GB | {:9.1} GB | {:9.1} GB | {:11.1} GB | {:4.0}%/{:4.0}% | Q:{} D:{}",
        size, our_quant_gbps, llama_quant_gbps, our_dequant_gbps, llama_dequant_gbps,
        quant_ratio, dequant_ratio, quant_match, dequant_match);
}

fn benchmark_q5_0(size: usize, runs: usize) {
    assert!(size % QK == 0);
    let nb = size / QK;

    let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut output_q5 = vec![BlockQ5_0 { d: 0, qh: [0u8; 4], qs: [0u8; QK / 2] }; nb];
    let mut output_f32 = vec![0.0f32; size];

    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q5_0(&input, &mut output_q5);
    }
    let our_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_quant_gbps = (size * 4) as f64 / our_quant_time / 1e9;

    let mut output_q5_llama = vec![BlockQ5_0 { d: 0, qh: [0u8; 4], qs: [0u8; QK / 2] }; nb];
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q5_0_llama(&input, &mut output_q5_llama);
    }
    let llama_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_quant_gbps = (size * 4) as f64 / llama_quant_time / 1e9;

    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q5_0(&output_q5, &mut output_f32);
    }
    let our_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_dequant_gbps = (size * 4) as f64 / our_dequant_time / 1e9;

    let mut output_f32_llama = vec![0.0f32; size];
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q5_0_llama(&output_q5_llama, &mut output_f32_llama);
    }
    let llama_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_dequant_gbps = (size * 4) as f64 / llama_dequant_time / 1e9;

    let quant_match = output_q5.iter().zip(output_q5_llama.iter()).all(|(a, b)| {
        a.d == b.d && a.qh == b.qh && a.qs == b.qs
    });
    let dequant_match = output_f32.iter().zip(output_f32_llama.iter()).all(|(a, b)| {
        (a - b).abs() < 1e-5
    });

    let quant_ratio = our_quant_gbps / llama_quant_gbps * 100.0;
    let dequant_ratio = our_dequant_gbps / llama_dequant_gbps * 100.0;

    println!("Q5_0   | {:5} | {:7.1} GB | {:9.1} GB | {:9.1} GB | {:11.1} GB | {:4.0}%/{:4.0}% | Q:{} D:{}",
        size, our_quant_gbps, llama_quant_gbps, our_dequant_gbps, llama_dequant_gbps,
        quant_ratio, dequant_ratio, quant_match, dequant_match);
}

fn benchmark_q5_1(size: usize, runs: usize) {
    assert!(size % QK == 0);
    let nb = size / QK;

    let input: Vec<f32> = (0..size).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut output_q5 = vec![BlockQ5_1 { d: 0, m: 0, qh: [0u8; 4], qs: [0u8; QK / 2] }; nb];
    let mut output_f32 = vec![0.0f32; size];

    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q5_1(&input, &mut output_q5);
    }
    let our_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_quant_gbps = (size * 4) as f64 / our_quant_time / 1e9;

    let mut output_q5_llama = vec![BlockQ5_1 { d: 0, m: 0, qh: [0u8; 4], qs: [0u8; QK / 2] }; nb];
    let start = Instant::now();
    for _ in 0..runs {
        quantize_row_q5_1_llama(&input, &mut output_q5_llama);
    }
    let llama_quant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_quant_gbps = (size * 4) as f64 / llama_quant_time / 1e9;

    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q5_1(&output_q5, &mut output_f32);
    }
    let our_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let our_dequant_gbps = (size * 4) as f64 / our_dequant_time / 1e9;

    let mut output_f32_llama = vec![0.0f32; size];
    let start = Instant::now();
    for _ in 0..runs {
        dequantize_row_q5_1_llama(&output_q5_llama, &mut output_f32_llama);
    }
    let llama_dequant_time = start.elapsed().as_secs_f64() / runs as f64;
    let llama_dequant_gbps = (size * 4) as f64 / llama_dequant_time / 1e9;

    let quant_match = output_q5.iter().zip(output_q5_llama.iter()).all(|(a, b)| {
        a.d == b.d && a.m == b.m && a.qh == b.qh && a.qs == b.qs
    });
    let dequant_match = output_f32.iter().zip(output_f32_llama.iter()).all(|(a, b)| {
        (a - b).abs() < 1e-5
    });

    let quant_ratio = our_quant_gbps / llama_quant_gbps * 100.0;
    let dequant_ratio = our_dequant_gbps / llama_dequant_gbps * 100.0;

    println!("Q5_1   | {:5} | {:7.1} GB | {:9.1} GB | {:9.1} GB | {:11.1} GB | {:4.0}%/{:4.0}% | Q:{} D:{}",
        size, our_quant_gbps, llama_quant_gbps, our_dequant_gbps, llama_dequant_gbps,
        quant_ratio, dequant_ratio, quant_match, dequant_match);
}

// ============================================================================
// llama.cpp reference implementations (scalar, from ggml-quants.c)
// ============================================================================

fn quantize_row_q8_0_llama(x: &[f32], y: &mut [BlockQ8_0]) {
    let nb = x.len() / QK;
    for i in 0..nb {
        let block = &mut y[i];

        // Find max absolute value
        let mut amax = 0.0f32;
        for j in 0..QK {
            amax = amax.max(x[i * QK + j].abs());
        }

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = fp32_to_fp16(d);

        for j in 0..QK {
            let v = x[i * QK + j] * id;
            block.qs[j] = v.round().clamp(-128.0, 127.0) as i8;
        }
    }
}

fn dequantize_row_q8_0_llama(x: &[BlockQ8_0], y: &mut [f32]) {
    let nb = x.len();
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..QK {
            y[i * QK + j] = x[i].qs[j] as f32 * d;
        }
    }
}

fn quantize_row_q4_0_llama(x: &[f32], y: &mut [BlockQ4_0]) {
    let nb = x.len() / QK;
    for i in 0..nb {
        let block = &mut y[i];

        // Find value with largest absolute value (llama.cpp exact match)
        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for j in 0..QK {
            let v = x[i * QK + j];
            if amax < v.abs() {
                amax = v.abs();
                max_val = v;
            }
        }

        let d = max_val / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = fp32_to_fp16(d);

        for j in 0..(QK / 2) {
            let v0 = (x[i * QK + j] * id + 8.5).min(15.0) as u32;
            let v1 = (x[i * QK + j + QK / 2] * id + 8.5).min(15.0) as u32;
            block.qs[j] = (v0 & 0xF) as u8 | ((v1 & 0xF) << 4) as u8;
        }
    }
}

fn dequantize_row_q4_0_llama(x: &[BlockQ4_0], y: &mut [f32]) {
    let nb = x.len();
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        for j in 0..(QK / 2) {
            let q0 = (x[i].qs[j] & 0x0F) as i32 - 8;
            let q1 = ((x[i].qs[j] >> 4) & 0x0F) as i32 - 8;
            y[i * QK + j] = q0 as f32 * d;
            y[i * QK + j + QK / 2] = q1 as f32 * d;
        }
    }
}

fn quantize_row_q4_1_llama(x: &[f32], y: &mut [BlockQ4_1]) {
    let nb = x.len() / QK;
    for i in 0..nb {
        let block = &mut y[i];

        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for j in 0..QK {
            min = min.min(x[i * QK + j]);
            max = max.max(x[i * QK + j]);
        }

        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = fp32_to_fp16(d);
        block.m = fp32_to_fp16(min);

        for j in 0..(QK / 2) {
            let v0 = ((x[i * QK + j] - min) * id).round().clamp(0.0, 15.0);
            let v1 = ((x[i * QK + j + QK / 2] - min) * id).round().clamp(0.0, 15.0);
            block.qs[j] = (v0 as u8) | ((v1 as u8) << 4);
        }
    }
}

fn dequantize_row_q4_1_llama(x: &[BlockQ4_1], y: &mut [f32]) {
    let nb = x.len();
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let m = fp16_to_fp32(x[i].m);
        for j in 0..(QK / 2) {
            let q0 = (x[i].qs[j] & 0x0F) as i32;
            let q1 = ((x[i].qs[j] >> 4) & 0x0F) as i32;
            y[i * QK + j] = q0 as f32 * d + m;
            y[i * QK + j + QK / 2] = q1 as f32 * d + m;
        }
    }
}

fn quantize_row_q5_0_llama(x: &[f32], y: &mut [BlockQ5_0]) {
    let nb = x.len() / QK;
    for i in 0..nb {
        let block = &mut y[i];

        // Find value with largest absolute value (llama.cpp exact match)
        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for j in 0..QK {
            let v = x[i * QK + j];
            if amax < v.abs() {
                amax = v.abs();
                max_val = v;
            }
        }

        let d = max_val / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = fp32_to_fp16(d);

        let mut qh = 0u32;
        for j in 0..(QK / 2) {
            let v0 = (x[i * QK + j] * id + 16.5).min(31.0) as u32;
            let v1 = (x[i * QK + j + QK / 2] * id + 16.5).min(31.0) as u32;

            block.qs[j] = (v0 & 0x0F) as u8 | ((v1 & 0x0F) << 4) as u8;

            if v0 & 0x10 != 0 { qh |= 1 << j; }
            if v1 & 0x10 != 0 { qh |= 1 << (j + 16); }
        }

        block.qh[0] = (qh & 0xFF) as u8;
        block.qh[1] = ((qh >> 8) & 0xFF) as u8;
        block.qh[2] = ((qh >> 16) & 0xFF) as u8;
        block.qh[3] = ((qh >> 24) & 0xFF) as u8;
    }
}

fn dequantize_row_q5_0_llama(x: &[BlockQ5_0], y: &mut [f32]) {
    let nb = x.len();
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);

        let qh = x[i].qh[0] as u32
               | ((x[i].qh[1] as u32) << 8)
               | ((x[i].qh[2] as u32) << 16)
               | ((x[i].qh[3] as u32) << 24);

        for j in 0..(QK / 2) {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;

            let q0 = ((x[i].qs[j] & 0x0F) as i32 | xh0 as i32) - 16;
            let q1 = (((x[i].qs[j] >> 4) & 0x0F) as i32 | xh1 as i32) - 16;

            y[i * QK + j] = q0 as f32 * d;
            y[i * QK + j + QK / 2] = q1 as f32 * d;
        }
    }
}

fn quantize_row_q5_1_llama(x: &[f32], y: &mut [BlockQ5_1]) {
    let nb = x.len() / QK;
    for i in 0..nb {
        let block = &mut y[i];

        let mut amax = f32::MIN;
        let mut min = f32::MAX;
        for j in 0..QK {
            amax = amax.max(x[i * QK + j]);
            min = min.min(x[i * QK + j]);
        }

        let d = (amax - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = fp32_to_fp16(d);
        block.m = fp32_to_fp16(min);

        let mut qh = 0u32;
        for j in 0..(QK / 2) {
            let v0 = ((x[i * QK + j] - min) * id).round().clamp(0.0, 31.0);
            let v1 = ((x[i * QK + j + QK / 2] - min) * id).round().clamp(0.0, 31.0);

            let q0 = v0 as u32;
            let q1 = v1 as u32;

            block.qs[j] = (q0 & 0x0F) as u8 | ((q1 & 0x0F) << 4) as u8;

            if q0 & 0x10 != 0 { qh |= 1 << j; }
            if q1 & 0x10 != 0 { qh |= 1 << (j + 16); }
        }

        block.qh[0] = (qh & 0xFF) as u8;
        block.qh[1] = ((qh >> 8) & 0xFF) as u8;
        block.qh[2] = ((qh >> 16) & 0xFF) as u8;
        block.qh[3] = ((qh >> 24) & 0xFF) as u8;
    }
}

fn dequantize_row_q5_1_llama(x: &[BlockQ5_1], y: &mut [f32]) {
    let nb = x.len();
    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d);
        let m = fp16_to_fp32(x[i].m);

        let qh = x[i].qh[0] as u32
               | ((x[i].qh[1] as u32) << 8)
               | ((x[i].qh[2] as u32) << 16)
               | ((x[i].qh[3] as u32) << 24);

        for j in 0..(QK / 2) {
            let xh0 = ((qh >> j) & 1) << 4;
            let xh1 = ((qh >> (j + 16)) & 1) << 4;

            let q0 = (x[i].qs[j] & 0x0F) as i32 | xh0 as i32;
            let q1 = ((x[i].qs[j] >> 4) & 0x0F) as i32 | xh1 as i32;

            y[i * QK + j] = q0 as f32 * d + m;
            y[i * QK + j + QK / 2] = q1 as f32 * d + m;
        }
    }
}
