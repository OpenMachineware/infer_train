// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Instrumented IQ1_M benchmark to find performance bottlenecks
// Measures each step's absolute time

use infer_train::quant::types::{BlockIQ1M, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::tables::IQ1S_GRID;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

const ITERATIONS: usize = 10000;
const N_BLOCKS: usize = 16;

fn main() {
    println!("=== IQ1_M Step-by-Step Instrumentation ===\n");

    // Generate test data
    let x: Vec<BlockIQ1M> = (0..N_BLOCKS)
        .map(|_| {
            let mut qs = [0u8; 54];
            let mut qh = [0u8; 32];
            let mut scales = [0u8; 8];
            for j in 0..54 {
                qs[j] = rand::random();
            }
            for j in 0..32 {
                qh[j] = rand::random();
            }
            for j in 0..8 {
                scales[j] = rand::random();
            }
            BlockIQ1M { scales, qs, qh }
        })
        .collect();

    let y: Vec<BlockQ8K> = (0..N_BLOCKS)
        .map(|_| BlockQ8K { d: 1.0, qs: [0; 256], bsums: [0; 16] })
        .collect();

    // Measure each step
    let mut total_grid_idx_ns = 0u64;
    let mut total_grid_load_ns = 0u64;
    let mut total_q8_load_ns = 0u64;
    let mut total_dotprod_ns = 0u64;
    let mut total_delta_ns = 0u64;

    for _ in 0..ITERATIONS {
        for i in 0..N_BLOCKS {
            let mut qs = x[i].qs.as_ptr();
            let mut qh = x[i].qh.as_ptr();

            // Step 1: Grid index computation
            let t1 = Instant::now();
            let qs_vals: [u8; 8] = [
                *qs.add(0),
                *qs.add(1),
                *qs.add(2),
                *qs.add(3),
                *qs.add(4),
                *qs.add(5),
                *qs.add(6),
                *qs.add(7),
            ];
            let qh_vals: [u8; 4] =
                [*qh.add(0), *qh.add(1), *qh.add(2), *qh.add(3)];

            let idx: [usize; 8] = [
                (qs_vals[0] as usize)
                    | (((qh_vals[0] as u32) << 8) & 0x700) as usize,
                (qs_vals[1] as usize)
                    | (((qh_vals[0] as u32) << 4) & 0x700) as usize,
                (qs_vals[2] as usize)
                    | (((qh_vals[1] as u32) << 8) & 0x700) as usize,
                (qs_vals[3] as usize)
                    | (((qh_vals[1] as u32) << 4) & 0x700) as usize,
                (qs_vals[4] as usize)
                    | (((qh_vals[2] as u32) << 8) & 0x700) as usize,
                (qs_vals[5] as usize)
                    | (((qh_vals[2] as u32) << 4) & 0x700) as usize,
                (qs_vals[6] as usize)
                    | (((qh_vals[3] as u32) << 8) & 0x700) as usize,
                (qs_vals[7] as usize)
                    | (((qh_vals[3] as u32) << 4) & 0x700) as usize,
            ];
            total_grid_idx_ns += t1.elapsed().as_nanos() as u64;

            // Step 2: Grid value loads
            let t2 = Instant::now();
            let grid_vals: [[i8; 8]; 8] = [
                IQ1S_GRID[idx[0]].to_le_bytes(),
                IQ1S_GRID[idx[1]].to_le_bytes(),
                IQ1S_GRID[idx[2]].to_le_bytes(),
                IQ1S_GRID[idx[3]].to_le_bytes(),
                IQ1S_GRID[idx[4]].to_le_bytes(),
                IQ1S_GRID[idx[5]].to_le_bytes(),
                IQ1S_GRID[idx[6]].to_le_bytes(),
                IQ1S_GRID[idx[7]].to_le_bytes(),
            ];
            std::hint::black_box(&grid_vals);
            total_grid_load_ns += t2.elapsed().as_nanos() as u64;

            // Step 3: Delta computation
            let t3 = Instant::now();
            let qh32 = (qh_vals[0] as u32)
                | ((qh_vals[1] as u32) << 8)
                | ((qh_vals[2] as u32) << 16)
                | ((qh_vals[3] as u32) << 24);
            let aux32 = ((qh32 >> 3) & 0x01010101) | ((qh32 >> 6) & 0x02020202);
            let aux8 = aux32.to_le_bytes();
            std::hint::black_box(&aux8);
            total_delta_ns += t3.elapsed().as_nanos() as u64;

            qs = qs.add(8);
            qh = qh.add(4);
        }
    }

    let total_ns = total_grid_idx_ns + total_grid_load_ns + total_delta_ns;
    let per_iter_ns = total_ns as f64 / ITERATIONS as f64 / N_BLOCKS as f64;

    println!("Step-by-step timing (per iteration):");
    println!(
        "  Grid index:    {:7.1} ns ({:.1}%)",
        total_grid_idx_ns as f64 / ITERATIONS as f64 / N_BLOCKS as f64,
        total_grid_idx_ns as f64 / total_ns as f64 * 100.0
    );
    println!(
        "  Grid load:     {:7.1} ns ({:.1}%)",
        total_grid_load_ns as f64 / ITERATIONS as f64 / N_BLOCKS as f64,
        total_grid_load_ns as f64 / total_ns as f64 * 100.0
    );
    println!(
        "  Delta calc:    {:7.1} ns ({:.1}%)",
        total_delta_ns as f64 / ITERATIONS as f64 / N_BLOCKS as f64,
        total_delta_ns as f64 / total_ns as f64 * 100.0
    );
    println!("  Total:         {:7.1} ns", per_iter_ns);
}
