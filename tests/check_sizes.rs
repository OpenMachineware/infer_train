// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check struct sizes and layouts

fn main() {
    use std::mem::size_of;

    // kv_cache types
    println!("=== kv_cache types ===");
    println!(
        "BlockQ4_0: {}",
        size_of::<infer_train::quant::kv_cache::BlockQ4_0>()
    );
    println!(
        "BlockQ4_1: {}",
        size_of::<infer_train::quant::kv_cache::BlockQ4_1>()
    );
    println!(
        "BlockQ5_0: {}",
        size_of::<infer_train::quant::kv_cache::BlockQ5_0>()
    );
    println!(
        "BlockQ5_1: {}",
        size_of::<infer_train::quant::kv_cache::BlockQ5_1>()
    );
    println!(
        "BlockQ8_0: {}",
        size_of::<infer_train::quant::kv_cache::BlockQ8_0>()
    );
    println!(
        "BlockQ8_1: {}",
        size_of::<infer_train::quant::kv_cache::BlockQ8_1>()
    );
    println!(
        "BlockIQ4NL: {}",
        size_of::<infer_train::quant::kv_cache::BlockIQ4NL>()
    );

    // types module types
    println!("\n=== types module ===");
    println!(
        "BlockQ4_0: {}",
        size_of::<infer_train::quant::types::BlockQ4_0>()
    );
    println!(
        "BlockQ4_1: {}",
        size_of::<infer_train::quant::types::BlockQ4_1>()
    );
    println!(
        "BlockQ5_0: {}",
        size_of::<infer_train::quant::types::BlockQ5_0>()
    );
    println!(
        "BlockQ5_1: {}",
        size_of::<infer_train::quant::types::BlockQ5_1>()
    );
    println!(
        "BlockQ8_0: {}",
        size_of::<infer_train::quant::types::BlockQ8_0>()
    );
    println!(
        "BlockQ8_1: {}",
        size_of::<infer_train::quant::types::BlockQ8_1>()
    );
    println!(
        "BlockIQ4NL: {}",
        size_of::<infer_train::quant::types::BlockIQ4NL>()
    );
}
