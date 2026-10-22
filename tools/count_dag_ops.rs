// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Count operation types in DAG

use std::collections::HashMap;

fn main() {
    println!("=== DAG Operation Count ===\n");

    println!("Operation types (by Discriminant):");
    println!("  0: EmbeddingLookup");
    println!("  1: RmsNorm");
    println!("  2: Gemv");
    println!("  3: FusedQKV");
    println!("  4: FusedGateUp");
    println!("  5: RoPE");
    println!("  6: FlashAttention");
    println!("  7: KVCacheUpdate");
    println!("  8: SiLU");
    println!("  9: Mul");
    println!("  10: Add");
    println!("  11: FusedSwiGLU");
    println!("  12: FusedNormLinear");
    println!("  13: FusedAddNorm");
    println!();

    println!("To get actual counts, run:");
    println!("  INFER_TRACE_DAG=1 cargo run --release --features metal --bin bench-decode -- <model.gguf> 2>&1 | grep '\\[TRACE\\] Node' | grep -o 'Discriminant([0-9]*)' | sort | uniq -c");
}