fn main() {
    // Link llama.cpp static library
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let native_dir = format!("{}/native", manifest_dir);
    println!("cargo:rustc-link-search=native={}", native_dir);
    println!("cargo:rustc-link-lib=static=llama_vecdot");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native/llama_vec_dot_wrapper.c");
}
