use criterion::{black_box, criterion_group, criterion_main, Criterion};
use infer_train::quant::vec_dot::scalar::fp32::vec_dot_fp32;
use infer_train::quant::vec_dot::scalar::fp16::vec_dot_fp16;
use infer_train::quant::vec_dot::scalar::bf16::vec_dot_bf16;
use infer_train::quant::vec_dot::scalar::q4_0::vec_dot_q4_0_q8_0;
use infer_train::quant::vec_dot::scalar::q8_0::vec_dot_q8_0_q8_0;
use infer_train::quant::vec_dot::scalar::q4_1::vec_dot_q4_1_q8_1;
use infer_train::quant::vec_dot::scalar::q2_k::vec_dot_q2_k_q8_k;
use infer_train::quant::vec_dot::scalar::q3_k::vec_dot_q3_k_q8_k;
use infer_train::quant::vec_dot::scalar::q4_k::vec_dot_q4_k_q8_k;
use infer_train::quant::vec_dot::scalar::q5_k::vec_dot_q5_k_q8_k;
use infer_train::quant::vec_dot::scalar::q6_k::vec_dot_q6_k_q8_k;

use infer_train::quant::types::{BlockQ4_0, BlockQ4_1, BlockQ8_0, BlockQ8_1, QK4_0, QK_K, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K};
use half::{f16, bf16};

// Generate test data
fn generate_fp32_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 * (i as f32).cos()).collect()
}

fn generate_fp16_data(n: usize) -> Vec<f16> {
    (0..n).map(|i| f16::from_f32(0.1 * (i as f32).cos())).collect()
}

fn generate_bf16_data(n: usize) -> Vec<bf16> {
    (0..n).map(|i| bf16::from_f32(0.1 * (i as f32).cos())).collect()
}

fn generate_q4_0_blocks(nb: usize) -> Vec<BlockQ4_0> {
    (0..nb).map(|i| {
        let d = 1.0f32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let v0 = ((i + j * 2) % 16) as i8 - 8;
            let v1 = ((i + j * 2 + 1) % 16) as i8 - 8;
            qs[j] = ((v0 + 8) as u8) | (((v1 + 8) as u8) << 4);
        }
        BlockQ4_0 { d: f16::from_f32(d).to_bits(), qs }
    }).collect()
}

fn generate_q8_0_blocks(nb: usize) -> Vec<BlockQ8_0> {
    (0..nb).map(|i| {
        let d = 1.0f32;
        let mut qs = [0i8; 32];
        for j in 0..32 {
            qs[j] = ((i + j) % 64 - 32) as i8;
        }
        BlockQ8_0 { d: f16::from_f32(d).to_bits(), qs }
    }).collect()
}

fn generate_q4_1_blocks(nb: usize) -> Vec<BlockQ4_1> {
    (0..nb).map(|i| {
        let d = 1.0f32;
        let m = 0.0f32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((i + j) % 16) as u8;
        }
        BlockQ4_1 { d: f16::from_f32(d).to_bits(), m: f16::from_f32(m).to_bits(), qs }
    }).collect()
}

fn generate_q8_1_blocks(nb: usize) -> Vec<BlockQ8_1> {
    (0..nb).map(|i| {
        let d = 1.0f32;
        let s = 0.0f32;
        let mut qs = [0i8; 32];
        for j in 0..32 {
            qs[j] = ((i + j) % 16) as i8;
        }
        BlockQ8_1 { d: f16::from_f32(d).to_bits(), s: f16::from_f32(s).to_bits(), qs }
    }).collect()
}

fn generate_q8_k_blocks(nb: usize) -> Vec<BlockQ8K> {
    (0..nb).map(|i| {
        let d = 1.0f32;
        let mut qs = [0i8; 256];
        for j in 0..256 {
            qs[j] = ((i + j) % 128 - 64) as i8;
        }
        let mut bsums = [0i16; 16];
        for j in 0..16 {
            let start = j * 16;
            let mut sum = 0i32;
            for k in 0..16 {
                sum += qs[start + k] as i32;
            }
            bsums[j] = sum as i16;
        }
        BlockQ8K { d, qs, bsums }
    }).collect()
}

fn generate_q2_k_blocks(nb: usize) -> Vec<BlockQ2K> {
    (0..nb).map(|i| {
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for j in 0..16 {
            scales[j] = ((i + j) % 8 + 1) as u8;
        }
        for j in 0..64 {
            qs[j] = ((i * 4 + j) % 256) as u8;
        }
        BlockQ2K {
            scales,
            qs,
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
        }
    }).collect()
}

fn generate_q3_k_blocks(nb: usize) -> Vec<BlockQ3K> {
    (0..nb).map(|i| {
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        let mut scales = [0u8; 12];
        for j in 0..32 {
            hmask[j] = ((i + j) % 256) as u8;
        }
        for j in 0..64 {
            qs[j] = ((i * 4 + j) % 256) as u8;
        }
        for j in 0..12 {
            scales[j] = ((i + j) % 64 + 16) as u8;
        }
        BlockQ3K { hmask, qs, scales, d: f16::from_f32(1.0).to_bits() }
    }).collect()
}

fn generate_q4_k_blocks(nb: usize) -> Vec<BlockQ4K> {
    (0..nb).map(|i| {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        for j in 0..12 {
            scales[j] = ((i + j) % 64 + 16) as u8;
        }
        for j in 0..128 {
            qs[j] = ((i * 2 + j) % 256) as u8;
        }
        BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        }
    }).collect()
}

fn generate_q5_k_blocks(nb: usize) -> Vec<BlockQ5K> {
    (0..nb).map(|i| {
        let mut scales = [0u8; 12];
        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        for j in 0..12 {
            scales[j] = ((i + j) % 64 + 16) as u8;
        }
        for j in 0..32 {
            qh[j] = ((i + j) % 256) as u8;
        }
        for j in 0..128 {
            qs[j] = ((i * 2 + j) % 256) as u8;
        }
        BlockQ5K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qh,
            qs,
        }
    }).collect()
}

fn generate_q6_k_blocks(nb: usize) -> Vec<BlockQ6K> {
    (0..nb).map(|i| {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for j in 0..128 {
            ql[j] = ((i * 2 + j) % 256) as u8;
        }
        for j in 0..64 {
            qh[j] = ((i + j) % 256) as u8;
        }
        for j in 0..16 {
            scales[j] = ((i + j) % 32 - 16) as i8;
        }
        BlockQ6K {
            ql,
            qh,
            scales,
            d: f16::from_f32(1.0).to_bits(),
        }
    }).collect()
}

fn bench_fp32(c: &mut Criterion) {
    let n = 1024 * 1024;  // 1M elements
    let a = generate_fp32_data(n);
    let b = generate_fp32_data(n);

    c.bench_function("fp32_vec_dot_1M", |bencher| {
        bencher.iter(|| vec_dot_fp32(black_box(&a), black_box(&b)))
    });
}

fn bench_fp16(c: &mut Criterion) {
    let n = 1024 * 1024;
    let a = generate_fp16_data(n);
    let b = generate_fp16_data(n);

    c.bench_function("fp16_vec_dot_1M", |bencher| {
        bencher.iter(|| vec_dot_fp16(black_box(&a), black_box(&b)))
    });
}

fn bench_bf16(c: &mut Criterion) {
    let n = 1024 * 1024;
    let a = generate_bf16_data(n);
    let b = generate_bf16_data(n);

    c.bench_function("bf16_vec_dot_1M", |bencher| {
        bencher.iter(|| vec_dot_bf16(black_box(&a), black_box(&b)))
    });
}

fn bench_q4_0(c: &mut Criterion) {
    let nb = 4096;  // 4096 blocks = 131072 elements
    let x = generate_q4_0_blocks(nb);
    let y = generate_q8_0_blocks(nb);
    let n = nb * QK4_0;

    c.bench_function("q4_0_q8_0_131k", |bencher| {
        bencher.iter(|| vec_dot_q4_0_q8_0(black_box(n), black_box(&x), black_box(&y)))
    });
}

fn bench_q8_0(c: &mut Criterion) {
    let nb = 4096;
    let x = generate_q8_0_blocks(nb);
    let y = generate_q8_0_blocks(nb);
    let n = nb * QK4_0;

    c.bench_function("q8_0_q8_0_131k", |bencher| {
        bencher.iter(|| vec_dot_q8_0_q8_0(black_box(n), black_box(&x), black_box(&y)))
    });
}

fn bench_q4_1(c: &mut Criterion) {
    let nb = 4096;
    let x = generate_q4_1_blocks(nb);
    let y = generate_q8_1_blocks(nb);
    let n = nb * QK4_0;

    c.bench_function("q4_1_q8_1_131k", |bencher| {
        bencher.iter(|| vec_dot_q4_1_q8_1(black_box(n), black_box(&x), black_box(&y)))
    });
}

fn bench_k_quant(c: &mut Criterion) {
    let nb = 256;  // 256 blocks = 65536 elements
    let n = nb * QK_K;

    let x_q2k = generate_q2_k_blocks(nb);
    let y_q8k = generate_q8_k_blocks(nb);
    c.bench_function("q2_k_q8_k_65k", |bencher| {
        bencher.iter(|| vec_dot_q2_k_q8_k(black_box(n), black_box(&x_q2k), black_box(&y_q8k)))
    });

    let x_q3k = generate_q3_k_blocks(nb);
    c.bench_function("q3_k_q8_k_65k", |bencher| {
        bencher.iter(|| vec_dot_q3_k_q8_k(black_box(n), black_box(&x_q3k), black_box(&y_q8k)))
    });

    let x_q4k = generate_q4_k_blocks(nb);
    c.bench_function("q4_k_q8_k_65k", |bencher| {
        bencher.iter(|| vec_dot_q4_k_q8_k(black_box(n), black_box(&x_q4k), black_box(&y_q8k)))
    });

    let x_q5k = generate_q5_k_blocks(nb);
    c.bench_function("q5_k_q8_k_65k", |bencher| {
        bencher.iter(|| vec_dot_q5_k_q8_k(black_box(n), black_box(&x_q5k), black_box(&y_q8k)))
    });

    let x_q6k = generate_q6_k_blocks(nb);
    c.bench_function("q6_k_q8_k_65k", |bencher| {
        bencher.iter(|| vec_dot_q6_k_q8_k(black_box(n), black_box(&x_q6k), black_box(&y_q8k)))
    });
}

criterion_group!(benches, bench_fp32, bench_fp16, bench_bf16, bench_q4_0, bench_q8_0, bench_q4_1, bench_k_quant);
criterion_main!(benches);
