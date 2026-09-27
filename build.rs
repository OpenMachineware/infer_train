fn main() {
    // Link native FP16 SIMD library
    println!("cargo:rustc-link-search=native={}", "native");
    println!("cargo:rustc-link-lib=static=fp16_attention");
    println!("cargo:rustc-link-lib=static=fp16_simd");
}
