// Build script for linking llama.cpp library
// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

fn main() {
    // Only link for llama.cpp benchmark binaries when llama-bench feature is enabled
    if std::env::var("CARGO_FEATURE_LLAMA_BENCH").is_ok() {
        // Link llama.cpp Metal library from tools directory
        println!("cargo:rustc-link-search=native=tools");
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
