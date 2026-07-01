// build.rs
use std::process::Command;
use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source_dir = manifest_dir.join("operators");
    let build_dir = source_dir.join("build");

    // 1. 确保 source_dir 存在
    if !source_dir.exists() {
        panic!("operators directory not found at {:?}", source_dir);
    }

    // 1. 创建 build 目录
    std::fs::create_dir_all(&build_dir).unwrap();

    // 2. 运行 meson setup
    println!("cargo:warning=Running meson setup...");
    let status = Command::new("meson")
        .arg("setup")
        .arg("--reconfigure")
        .arg(&build_dir)
        .arg(&source_dir)
        .status()
        .unwrap_or_else(|_| {
            panic!("Failed to run meson. Is meson installed?");
        });

    if !status.success() {
        panic!("Meson setup failed");
    }

    // 3. 运行 meson compile
    let status = Command::new("meson")
        .arg("compile")
        .arg("-C")
        .arg(&build_dir)
        .status()
        .unwrap();

    if !status.success() {
        panic!("Meson compile failed");
    }

    // 4. 告诉 cargo 去哪里找静态库
    let _lib_dir = build_dir.join("libinfer_train_cpp.a");
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=infer_train_cpp");
    println!("cargo:rerun-if-changed=cpp/");

    // ============================================================
    // 新增：链接 C++ 标准库
    // ============================================================
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=c++");   // macOS: libc++
        println!("cargo:rustc-link-lib=c++abi");
    }
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=stdc++"); // Linux: libstdc++
    }
    #[cfg(target_os = "windows")]
    {
        // Windows 通常不需要额外链接
    }

    println!("cargo:rerun-if-changed=operators/");
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:warning=C++ operators compiled successfully!");
}
