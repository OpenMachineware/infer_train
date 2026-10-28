// SPDX-License-Identifier: AGPL-3.0-or-later
// Test w3 (up projection) GEMV

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let layer = &ctx.model.layers[0];
    let w1 = layer.w1.as_ref().unwrap(); // gate
    let w3 = layer.w3.as_ref().unwrap(); // up

    println!("w1 (gate): shape={:?} dtype={:?}", w1.shape, w1.dtype);
    println!("w3 (up): shape={:?} dtype={:?}", w3.shape, w3.dtype);

    let hidden_size = 1536;
    let intermediate_size = w1.shape[1];

    // Test input (normalized hidden)
    let test_input = vec![0.2865f32, -0.4736, 0.1866, 0.0553, -0.0055];
    let test_input_padded: Vec<f32> = {
        let mut v = vec![0.0f32; hidden_size];
        v[..5].copy_from_slice(&test_input);
        v
    };

    // Test gate projection
    let gate =
        ctx.model.gemv_impl(&test_input_padded, w1, intermediate_size).unwrap();
    println!(
        "\nGate output (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        gate[0], gate[1], gate[2], gate[3], gate[4]
    );

    // Test up projection
    let up =
        ctx.model.gemv_impl(&test_input_padded, w3, intermediate_size).unwrap();
    println!(
        "Up output (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        up[0], up[1], up[2], up[3], up[4]
    );

    // Check if up is all zeros
    let up_zeros = up.iter().all(|&x| x == 0.0);
    println!("Up all zeros: {}", up_zeros);

    // Check w3 data
    println!("\nw3 bytes size: {} bytes", w3.bytes().len());
    let first_bytes = &w3.bytes()[..std::cmp::min(20, w3.bytes().len())];
    println!("w3 first 20 bytes: {:?}", first_bytes);
}
