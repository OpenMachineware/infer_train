fn main() {
    // Link llama.cpp CPU library for benchmarking
    println!("cargo:rustc-link-search=native=llama.cpp-0.5.0/build/bin");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");
}
