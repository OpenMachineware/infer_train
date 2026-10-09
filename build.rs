// Build script for linking llama.cpp library
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    // Only link for the benchmark binary
    if std::env::var("CARGO_BIN_NAME").unwrap_or_default()
        == "bench_flash_attn_vs_llama_fair"
    {
        // Link llama.cpp Metal library from the correct build directory
        println!("cargo:rustc-link-search=native=tools");
        println!(
            "cargo:rustc-link-search=native=tools/iq2_xxs_llama_dump/build/bin"
        );
        println!("cargo:rustc-link-lib=dylib=llama_fa_bench");
        println!("cargo:rustc-link-lib=dylib=ggml-metal");
        println!("cargo:rustc-link-lib=dylib=ggml-cpu");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
        println!("cargo:rustc-link-lib=dylib=ggml");

        // Link Metal frameworks
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
