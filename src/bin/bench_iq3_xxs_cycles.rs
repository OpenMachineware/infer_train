// Use cycle counter for precise step-by-step measurement
// Avoids Instant::now() overhead

use std::arch::aarch64::*;
use std::hint::black_box;

use infer_train::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::fp::fp16_to_fp32;
use infer_train::quant::vec_dot::tables::{IQ3XXS_GRID, KEVEN_SIGNS_Q2XS};

const ITERATIONS: usize = 10000;
const N_BLOCKS: usize = 16;

fn main() {
    println!("=== IQ3_XXS Cycle-Counter Benchmark ===\n");

    // Generate test data
    let x: Vec<BlockIQ3XXS> =
        (0..N_BLOCKS).map(|_| BlockIQ3XXS { d: 15360, qs: [0; 96] }).collect();
    let y: Vec<BlockQ8K> = (0..N_BLOCKS)
        .map(|_| BlockQ8K { d: 1.0, qs: [0; 256], bsums: [0; 16] })
        .collect();

    unsafe {
        // Warm up
        for _ in 0..100 {
            let _ = instrumented_rust(N_BLOCKS * QK_K, &x, &y);
        }

        // Use PMU cycle counter
        enable_pmu();

        let mut total_cycles = 0u64;
        let mut q8_load_cycles = 0u64;
        let mut aux32_load_cycles = 0u64;
        let mut grid_cycles = 0u64;
        let mut sign_cycles = 0u64;
        let mut dot_cycles = 0u64;
        let mut accum_cycles = 0u64;

        for _ in 0..ITERATIONS {
            let nb = N_BLOCKS;
            let mut sum = 0.0f32;
            let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;
            let grid = IQ3XXS_GRID.as_ptr();
            let half = 0.5f32;
            let zero = vdupq_n_s32(0);

            for i in 0..nb {
                let d = fp16_to_fp32(x[i].d) * y[i].d;
                let mut q3 = x[i].qs.as_ptr();
                let mut gas = q3.add(QK_K / 4);
                let mut q8 = y[i].qs.as_ptr();

                let mut sumf1 = 0.0f32;
                let mut sumf2 = 0.0f32;

                for _ in 0..4 {
                    let t0 = read_pmccntr();

                    // Step 1: Q8 load
                    let q8b = vld1q_s8_x4(q8);
                    q8 = q8.add(64);
                    let t1 = read_pmccntr();

                    // Step 2: aux32 load
                    let aux32_0 = *(gas as *const u32);
                    let aux32_1 = *(gas.add(4) as *const u32);
                    gas = gas.add(8);
                    let t2 = read_pmccntr();

                    // Step 3: Grid lookup
                    let v0 = vld1q_u32_from_scalars(
                        *grid.add(*q3.add(0) as usize),
                        *grid.add(*q3.add(1) as usize),
                        *grid.add(*q3.add(2) as usize),
                        *grid.add(*q3.add(3) as usize),
                    );
                    let v1 = vld1q_u32_from_scalars(
                        *grid.add(*q3.add(4) as usize),
                        *grid.add(*q3.add(5) as usize),
                        *grid.add(*q3.add(6) as usize),
                        *grid.add(*q3.add(7) as usize),
                    );
                    let v2 = vld1q_u32_from_scalars(
                        *grid.add(*q3.add(8) as usize),
                        *grid.add(*q3.add(9) as usize),
                        *grid.add(*q3.add(10) as usize),
                        *grid.add(*q3.add(11) as usize),
                    );
                    let v3 = vld1q_u32_from_scalars(
                        *grid.add(*q3.add(12) as usize),
                        *grid.add(*q3.add(13) as usize),
                        *grid.add(*q3.add(14) as usize),
                        *grid.add(*q3.add(15) as usize),
                    );
                    q3 = q3.add(16);
                    let t3 = read_pmccntr();

                    // Step 4: Sign application
                    let q3s_0 = vmulq_s8(
                        vcombine_s8(
                            vld1_s8(
                                signs64
                                    .add(((aux32_0 >> 0) & 127) as usize * 8),
                            ),
                            vld1_s8(
                                signs64
                                    .add(((aux32_0 >> 7) & 127) as usize * 8),
                            ),
                        ),
                        vreinterpretq_s8_u32(v0),
                    );
                    let q3s_1 = vmulq_s8(
                        vcombine_s8(
                            vld1_s8(
                                signs64
                                    .add(((aux32_0 >> 14) & 127) as usize * 8),
                            ),
                            vld1_s8(
                                signs64
                                    .add(((aux32_0 >> 21) & 127) as usize * 8),
                            ),
                        ),
                        vreinterpretq_s8_u32(v1),
                    );
                    let q3s_2 = vmulq_s8(
                        vcombine_s8(
                            vld1_s8(
                                signs64
                                    .add(((aux32_1 >> 0) & 127) as usize * 8),
                            ),
                            vld1_s8(
                                signs64
                                    .add(((aux32_1 >> 7) & 127) as usize * 8),
                            ),
                        ),
                        vreinterpretq_s8_u32(v2),
                    );
                    let q3s_3 = vmulq_s8(
                        vcombine_s8(
                            vld1_s8(
                                signs64
                                    .add(((aux32_1 >> 14) & 127) as usize * 8),
                            ),
                            vld1_s8(
                                signs64
                                    .add(((aux32_1 >> 21) & 127) as usize * 8),
                            ),
                        ),
                        vreinterpretq_s8_u32(v3),
                    );
                    let t4 = read_pmccntr();

                    // Step 5: Dot products
                    let p1 = vdotq_s32_manual(
                        vdotq_s32_manual(zero, q3s_0, q8b.0),
                        q3s_1,
                        q8b.1,
                    );
                    let p2 = vdotq_s32_manual(
                        vdotq_s32_manual(zero, q3s_2, q8b.2),
                        q3s_3,
                        q8b.3,
                    );
                    let t5 = read_pmccntr();

                    // Step 6: Accumulation
                    sumf1 +=
                        vaddvq_s32(p1) as f32 * (half + (aux32_0 >> 28) as f32);
                    sumf2 +=
                        vaddvq_s32(p2) as f32 * (half + (aux32_1 >> 28) as f32);
                    let t6 = read_pmccntr();

                    q8_load_cycles += t1 - t0;
                    aux32_load_cycles += t2 - t1;
                    grid_cycles += t3 - t2;
                    sign_cycles += t4 - t3;
                    dot_cycles += t5 - t4;
                    accum_cycles += t6 - t5;
                    total_cycles += t6 - t0;
                }

                sum += d * (sumf1 + sumf2) * half;
            }
        }

        let total_ops = ITERATIONS * N_BLOCKS * 4;

        println!("Rust Implementation (cycles per inner loop iteration):");
        println!(
            "  Q8 load:        {:6.1} cycles",
            q8_load_cycles as f64 / total_ops as f64
        );
        println!(
            "  aux32 load:     {:6.1} cycles",
            aux32_load_cycles as f64 / total_ops as f64
        );
        println!(
            "  Grid lookup:    {:6.1} cycles",
            grid_cycles as f64 / total_ops as f64
        );
        println!(
            "  Sign apply:     {:6.1} cycles",
            sign_cycles as f64 / total_ops as f64
        );
        println!(
            "  Dot product:    {:6.1} cycles",
            dot_cycles as f64 / total_ops as f64
        );
        println!(
            "  Accumulation:   {:6.1} cycles",
            accum_cycles as f64 / total_ops as f64
        );
        println!("  ----------------------------------------");
        println!(
            "  Total:          {:6.1} cycles",
            total_cycles as f64 / total_ops as f64
        );

        // Now measure llama.cpp
        println!("\n=== llama.cpp Reference ===");
        measure_llama_cpp_cycles();
    }
}

unsafe fn instrumented_rust(
    n: usize,
    x: &[BlockIQ3XXS],
    y: &[BlockQ8K],
) -> f32 {
    let nb = n / QK_K;
    let mut sum = 0.0f32;
    let signs64 = KEVEN_SIGNS_Q2XS.as_ptr() as *const i8;
    let grid = IQ3XXS_GRID.as_ptr();
    let half = 0.5f32;
    let zero = vdupq_n_s32(0);

    for i in 0..nb {
        let d = fp16_to_fp32(x[i].d) * y[i].d;
        let mut q3 = x[i].qs.as_ptr();
        let mut gas = q3.add(QK_K / 4);
        let mut q8 = y[i].qs.as_ptr();

        let mut sumf1 = 0.0f32;
        let mut sumf2 = 0.0f32;

        for _ in 0..4 {
            let q8b = vld1q_s8_x4(q8);
            q8 = q8.add(64);

            let aux32_0 = *(gas as *const u32);
            let aux32_1 = *(gas.add(4) as *const u32);
            gas = gas.add(8);

            let v0 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(0) as usize),
                *grid.add(*q3.add(1) as usize),
                *grid.add(*q3.add(2) as usize),
                *grid.add(*q3.add(3) as usize),
            );
            let v1 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(4) as usize),
                *grid.add(*q3.add(5) as usize),
                *grid.add(*q3.add(6) as usize),
                *grid.add(*q3.add(7) as usize),
            );
            let v2 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(8) as usize),
                *grid.add(*q3.add(9) as usize),
                *grid.add(*q3.add(10) as usize),
                *grid.add(*q3.add(11) as usize),
            );
            let v3 = vld1q_u32_from_scalars(
                *grid.add(*q3.add(12) as usize),
                *grid.add(*q3.add(13) as usize),
                *grid.add(*q3.add(14) as usize),
                *grid.add(*q3.add(15) as usize),
            );
            q3 = q3.add(16);

            let q3s_0 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs64.add(((aux32_0 >> 0) & 127) as usize * 8)),
                    vld1_s8(signs64.add(((aux32_0 >> 7) & 127) as usize * 8)),
                ),
                vreinterpretq_s8_u32(v0),
            );
            let q3s_1 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs64.add(((aux32_0 >> 14) & 127) as usize * 8)),
                    vld1_s8(signs64.add(((aux32_0 >> 21) & 127) as usize * 8)),
                ),
                vreinterpretq_s8_u32(v1),
            );
            let q3s_2 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs64.add(((aux32_1 >> 0) & 127) as usize * 8)),
                    vld1_s8(signs64.add(((aux32_1 >> 7) & 127) as usize * 8)),
                ),
                vreinterpretq_s8_u32(v2),
            );
            let q3s_3 = vmulq_s8(
                vcombine_s8(
                    vld1_s8(signs64.add(((aux32_1 >> 14) & 127) as usize * 8)),
                    vld1_s8(signs64.add(((aux32_1 >> 21) & 127) as usize * 8)),
                ),
                vreinterpretq_s8_u32(v3),
            );

            let p1 = vdotq_s32_manual(
                vdotq_s32_manual(zero, q3s_0, q8b.0),
                q3s_1,
                q8b.1,
            );
            let p2 = vdotq_s32_manual(
                vdotq_s32_manual(zero, q3s_2, q8b.2),
                q3s_3,
                q8b.3,
            );

            sumf1 += vaddvq_s32(p1) as f32 * (half + (aux32_0 >> 28) as f32);
            sumf2 += vaddvq_s32(p2) as f32 * (half + (aux32_1 >> 28) as f32);
        }

        sum += d * (sumf1 + sumf2) * half;
    }

    sum
}

fn measure_llama_cpp_cycles() {
    let c_code = r#"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <arm_neon.h>

#define QK_K 256
#define ITERATIONS 10000
#define N_BLOCKS 16

extern const uint32_t iq3xxs_grid[256];
extern const int8_t keven_signs_q2xs[128];

static inline uint32x4_t ggml_vld1q_u32(uint32_t w, uint32_t x, uint32_t y, uint32_t z) {
    return (uint32x4_t){ w, x, y, z };
}

static inline uint64_t read_pmccntr() {
    uint64_t val;
    __asm__ volatile("mrs %0, pmccntr_el0" : "=r"(val));
    return val;
}

int main() {
    // Enable PMU
    __asm__ volatile("msr pmcr_el0, %0" :: "r"(1 | (1 << 1) | (1 << 2)));
    __asm__ volatile("msr pmcntenset_el0, %0" :: "r"(1 << 31));

    struct { uint16_t d; uint8_t qs[96]; } x[N_BLOCKS];
    struct { float d; int8_t qs[256]; uint16_t bsums[16]; } y[N_BLOCKS];

    for (int i = 0; i < N_BLOCKS; i++) {
        x[i].d = 15360;
        memset(x[i].qs, 0, 96);
        y[i].d = 1.0f;
        memset(y[i].qs, 0, 256);
    }

    // Warm up
    for (int iter = 0; iter < 100; iter++) {
        volatile float result = 0;
        for (int i = 0; i < N_BLOCKS; i++) {
            float d = (float)x[i].d * y[i].d;
            const uint8_t * q3 = x[i].qs;
            const uint8_t * gas = x[i].qs + QK_K/4;
            const int8_t * q8 = y[i].qs;
            float sumf1 = 0, sumf2 = 0;

            for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
                int8x16x4_t q8b = vld1q_s8_x4(q8); q8 += 64;
                uint32_t aux32[2];
                memcpy(aux32, gas, 8); gas += 8;

                const uint32x4_t aux32x4_0 = ggml_vld1q_u32(iq3xxs_grid[q3[0]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[3]]);
                const uint32x4_t aux32x4_1 = ggml_vld1q_u32(iq3xxs_grid[q3[4]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[7]]);
                const uint32x4_t aux32x4_2 = ggml_vld1q_u32(iq3xxs_grid[q3[8]], iq3xxs_grid[q3[9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]);
                const uint32x4_t aux32x4_3 = ggml_vld1q_u32(iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]);
                q3 += 16;

                int8x16x4_t q3s;
                q3s.val[0] = vmulq_s8(vcombine_s8(vld1_s8(keven_signs_q2xs + ((aux32[0] >> 0) & 127)), vld1_s8(keven_signs_q2xs + ((aux32[0] >> 7) & 127))), vreinterpretq_s8_u32(aux32x4_0));
                q3s.val[1] = vmulq_s8(vcombine_s8(vld1_s8(keven_signs_q2xs + ((aux32[0] >> 14) & 127)), vld1_s8(keven_signs_q2xs + ((aux32[0] >> 21) & 127))), vreinterpretq_s8_u32(aux32x4_1));
                q3s.val[2] = vmulq_s8(vcombine_s8(vld1_s8(keven_signs_q2xs + ((aux32[1] >> 0) & 127)), vld1_s8(keven_signs_q2xs + ((aux32[1] >> 7) & 127))), vreinterpretq_s8_u32(aux32x4_2));
                q3s.val[3] = vmulq_s8(vcombine_s8(vld1_s8(keven_signs_q2xs + ((aux32[1] >> 14) & 127)), vld1_s8(keven_signs_q2xs + ((aux32[1] >> 21) & 127))), vreinterpretq_s8_u32(aux32x4_3));

                const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q3s.val[0], q8b.val[0]), q3s.val[1], q8b.val[1]);
                const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q3s.val[2], q8b.val[2]), q3s.val[3], q8b.val[3]);

                sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[0] >> 28));
                sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[1] >> 28));
            }
            result = d * (sumf1 + sumf2) * 0.5f;
        }
    }

    // Instrumented measurement
    uint64_t total_cycles = 0;
    uint64_t q8_cycles = 0, aux_cycles = 0, grid_cycles = 0, sign_cycles = 0, dot_cycles = 0, accum_cycles = 0;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < N_BLOCKS; i++) {
            const uint8_t * q3 = x[i].qs;
            const uint8_t * gas = x[i].qs + QK_K/4;
            const int8_t * q8 = y[i].qs;

            for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
                uint64_t t0 = read_pmccntr();
                int8x16x4_t q8b = vld1q_s8_x4(q8); q8 += 64;
                uint64_t t1 = read_pmccntr();

                uint32_t aux32[2];
                memcpy(aux32, gas, 8); gas += 8;
                uint64_t t2 = read_pmccntr();

                const uint32x4_t aux32x4_0 = ggml_vld1q_u32(iq3xxs_grid[q3[0]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[3]]);
                const uint32x4_t aux32x4_1 = ggml_vld1q_u32(iq3xxs_grid[q3[4]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[7]]);
                const uint32x4_t aux32x4_2 = ggml_vld1q_u32(iq3xxs_grid[q3[8]], iq3xxs_grid[q3[9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]);
                const uint32x4_t aux32x4_3 = ggml_vld1q_u32(iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]);
                q3 += 16;
                uint64_t t3 = read_pmccntr();

                int8x16x4_t q3s;
                q3s.val[0] = vmulq_s8(vcombine_s8(vld1_s8(keven_signs_q2xs + ((aux32[0] >> 0) & 127)), vld1_s8(keven_signs_q2xs + ((aux32[0] >> 7) & 127))), vreinterpretq_s8_u32(aux32x4_0));
                q3s.val[1] = vmulq_s8(vcombine_s8(vld1_s8(keven_signs_q2xs + ((aux32[0] >> 14) & 127)), vld1_s8(keven_signs_q2xs + ((aux32[0] >> 21) & 127))), vreinterpretq_s8_u32(aux32x4_1));
                q3s.val[2] = vmulq_s8(vcombine_s8(vld1_s8(keven_signs_q2xs + ((aux32[1] >> 0) & 127)), vld1_s8(keven_signs_q2xs + ((aux32[1] >> 7) & 127))), vreinterpretq_s8_u32(aux32x4_2));
                q3s.val[3] = vmulq_s8(vcombine_s8(vld1_s8(keven_signs_q2xs + ((aux32[1] >> 14) & 127)), vld1_s8(keven_signs_q2xs + ((aux32[1] >> 21) & 127))), vreinterpretq_s8_u32(aux32x4_3));
                uint64_t t4 = read_pmccntr();

                const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q3s.val[0], q8b.val[0]), q3s.val[1], q8b.val[1]);
                const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q3s.val[2], q8b.val[2]), q3s.val[3], q8b.val[3]);
                uint64_t t5 = read_pmccntr();

                float sumf1 = vaddvq_s32(p1) * (0.5f + (aux32[0] >> 28));
                float sumf2 = vaddvq_s32(p2) * (0.5f + (aux32[1] >> 28));
                (void)sumf1; (void)sumf2;
                uint64_t t6 = read_pmccntr();

                q8_cycles += t1 - t0;
                aux_cycles += t2 - t1;
                grid_cycles += t3 - t2;
                sign_cycles += t4 - t3;
                dot_cycles += t5 - t4;
                accum_cycles += t6 - t5;
                total_cycles += t6 - t0;
            }
        }
    }

    double total_ops = ITERATIONS * N_BLOCKS * 4.0;
    printf("llama.cpp Implementation (cycles per inner loop iteration):\n");
    printf("  Q8 load:        %6.1f cycles\n", q8_cycles / total_ops);
    printf("  aux32 load:     %6.1f cycles\n", aux_cycles / total_ops);
    printf("  Grid lookup:    %6.1f cycles\n", grid_cycles / total_ops);
    printf("  Sign apply:     %6.1f cycles\n", sign_cycles / total_ops);
    printf("  Dot product:    %6.1f cycles\n", dot_cycles / total_ops);
    printf("  Accumulation:   %6.1f cycles\n", accum_cycles / total_ops);
    printf("  ----------------------------------------\n");
    printf("  Total:          %6.1f cycles\n", total_cycles / total_ops);

    return 0;
}
"#;

    std::fs::write("/tmp/iq3_xxs_cycles.c", c_code).unwrap();

    let compile_output = std::process::Command::new("clang")
        .args([
            "-O3",
            "-mcpu=apple-m1",
            "-I",
            "llama.cpp-0.5.0/ggml/src",
            "-I",
            "llama.cpp-0.5.0/ggml/src/ggml-cpu",
            "-I",
            "llama.cpp-0.5.0/ggml/src/ggml-cpu/arch/arm",
            "-I",
            "llama.cpp-0.5.0/ggml/include",
            "-o",
            "/tmp/iq3_xxs_cycles",
            "/tmp/iq3_xxs_cycles.c",
            "native/llama_vec_dot_wrapper.o",
        ])
        .current_dir("/Users/jia/Desktop/infer_train")
        .output()
        .expect("Failed to compile C code");

    if !compile_output.status.success() {
        println!("Compilation failed:");
        println!("{}", String::from_utf8_lossy(&compile_output.stderr));
        return;
    }

    let run_output = std::process::Command::new("/tmp/iq3_xxs_cycles")
        .output()
        .expect("Failed to run C code");

    println!("{}", String::from_utf8_lossy(&run_output.stdout));
}

// PMU helpers
unsafe fn enable_pmu() {
    use std::arch::asm;
    // Enable PMU cycle counter
    // PMCR_EL0: Enable bit (0), Cycle counter reset (1), Event counter reset (2)
    asm!(
        "msr pmcr_el0, {0}",
        in(reg) 0b111u64,
    );
    // Enable cycle counter
    asm!(
        "msr pmcntenset_el0, {0}",
        in(reg) (1u64 << 31),
    );
}

unsafe fn read_pmccntr() -> u64 {
    use std::arch::asm;
    let val: u64;
    asm!(
        "mrs {0}, pmccntr_el0",
        out(reg) val,
    );
    val
}

#[target_feature(enable = "neon")]
unsafe fn vld1q_u32_from_scalars(w: u32, x: u32, y: u32, z: u32) -> uint32x4_t {
    vcombine_u32(
        vreinterpret_u32_u64(vcreate_u64((w as u64) | ((x as u64) << 32))),
        vreinterpret_u32_u64(vcreate_u64((y as u64) | ((z as u64) << 32))),
    )
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn vdotq_s32_manual(
    acc: int32x4_t,
    a: int8x16_t,
    b: int8x16_t,
) -> int32x4_t {
    use std::arch::asm;
    let mut result = acc;
    asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, readonly, preserves_flags)
    );
    result
}
