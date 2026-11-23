// Compare layer output with llama.cpp reference
// This helps identify which layer is producing wrong results

use std::process::Command;

fn main() {
    println!("=== Layer-by-layer comparison ===\n");

    // Check if llama.cpp is available
    let llama_path = "/Users/jia/Desktop/infer_train/llama.cpp-1.0.0/build/bin/llama-cli";

    let output = Command::new(llama_path)
        .arg("--model")
        .arg("/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf")
        .arg("--prompt")
        .arg("Hello")
        .arg("--n-predict")
        .arg("10")
        .arg("--temp")
        .arg("0.0")
        .arg("--log-disable")
        .output();

    match output {
        Ok(output) => {
            println!("llama.cpp output:");
            println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
            println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        }
        Err(e) => {
            println!("Failed to run llama.cpp: {}", e);
            println!("Make sure llama.cpp is compiled at: {}", llama_path);
        }
    }

    // Also check quantization types used
    println!("\n=== Checking Q5_K and Q6_K weights ===");
    println!("Q5_K is used for attention Q/K weights");
    println!("Q6_K is used for attention V and FFN down weights");
    println!("Q4_K is used for other weights");
}
