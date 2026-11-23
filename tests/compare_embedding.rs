// Compare embedding between Rust and llama.cpp
// Run: cargo test compare_embedding -- --nocapture

use infer_train::backend::arm::dequant::dequant_q6_k;
use infer_train::model::GgufFile;
use infer_train::types::{BlockQ6_K, QK_K};

#[test]
fn compare_embedding() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let token_id: usize = 9707;

    // Load GGUF
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Find embedding tensor
    let emb_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("Embedding tensor not found");

    println!(
        "Embedding tensor: dims={:?}, strides={:?}, offset={}",
        emb_tensor.dims, emb_tensor.strides, emb_tensor.offset
    );
    println!("Quant type: {:?}", emb_tensor.quant_type);

    let hidden_size = emb_tensor.dims[0] as usize;
    let vocab_size = emb_tensor.dims[1] as usize;
    let tensor_offset = emb_tensor.offset as usize;

    println!("hidden_size={}, vocab_size={}", hidden_size, vocab_size);

    // Calculate row offset
    let nb = (hidden_size + QK_K - 1) / QK_K;
    let block_size = std::mem::size_of::<BlockQ6_K>();

    // Read embedding data
    let emb_data = &gguf.data[tensor_offset..];

    // Different offset calculation methods
    let stride = emb_tensor.strides[1] as usize;
    let method1_offset = token_id * stride;
    let method2_offset = token_id * hidden_size; // If tensor is stored row-major

    // Bounds check
    if method1_offset + nb * block_size > emb_data.len() {
        println!("WARNING: method1_offset out of bounds");
        return;
    }

    println!("\nOffset calculations:");
    println!("  stride[1] = {}", stride);
    println!("  Method 1 (token_id * stride): {}", method1_offset);
    println!("  Method 2 (token_id * hidden_size): {}", method2_offset);

    // Check raw bytes at method 1 offset
    println!("\nRaw bytes at offset {} (method 1):", method1_offset);
    println!(
        "  First 32 bytes: {:02x?}",
        &emb_data[method1_offset..method1_offset + 32]
    );

    // Read block at method 1 offset
    let block1: &BlockQ6_K = unsafe {
        &*(emb_data.as_ptr().add(method1_offset) as *const BlockQ6_K)
    };
    println!(
        "  Block d = 0x{:04x} = {:.6}",
        block1.d,
        half::f16::from_bits(block1.d).to_f32()
    );
    println!("  Block scales[:8] = {:?}", &block1.scales[..8]);

    // Dequantize method 1
    let mut emb1 = vec![0.0f32; hidden_size];
    for i in 0..nb {
        let block_offset = method1_offset + i * block_size;
        let block_bytes = &emb_data[block_offset..block_offset + block_size];
        dequant_q6_k(
            block_bytes,
            &mut emb1[i * QK_K..],
            QK_K.min(hidden_size - i * QK_K),
        );
    }

    // Calculate stats for method 1
    let min1 = emb1.iter().cloned().fold(f32::INFINITY, f32::min);
    let max1 = emb1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean1: f32 = emb1.iter().sum::<f32>() / emb1.len() as f32;

    println!("\nMethod 1 embedding (token_id * stride):");
    println!("  min={:.6}, max={:.6}, mean={:.6}", min1, max1, mean1);
    println!("  First 10: {:?}", &emb1[..10]);

    // Try method 2: direct offset without stride
    println!("\n--- Trying alternative offset calculations ---");

    // Method 3: blocks are stored contiguously, so row * nb_blocks * block_size
    let method3_offset = token_id * nb * block_size;
    println!(
        "Method 3 (token_id * nb * block_size): offset={}",
        method3_offset
    );

    if method3_offset + nb * block_size <= emb_data.len() {
        let block3: &BlockQ6_K = unsafe {
            &*(emb_data.as_ptr().add(method3_offset) as *const BlockQ6_K)
        };
        println!(
            "  Block d = 0x{:04x} = {:.6}",
            block3.d,
            half::f16::from_bits(block3.d).to_f32()
        );

        let mut emb3 = vec![0.0f32; hidden_size];
        for i in 0..nb {
            let block_offset = method3_offset + i * block_size;
            let block_bytes =
                &emb_data[block_offset..block_offset + block_size];
            dequant_q6_k(
                block_bytes,
                &mut emb3[i * QK_K..],
                QK_K.min(hidden_size - i * QK_K),
            );
        }

        let min3 = emb3.iter().cloned().fold(f32::INFINITY, f32::min);
        let max3 = emb3.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean3: f32 = emb3.iter().sum::<f32>() / emb3.len() as f32;

        println!("  min={:.6}, max={:.6}, mean={:.6}", min3, max3, mean3);
        println!("  First 10: {:?}", &emb3[..10]);
    }

    // Expected values from llama.cpp
    println!("\n=== Expected from llama.cpp (token 9707) ===");
    println!("  min=-55.761356, max=20.930227, mean=-0.213288");
    println!(
        "  First 10: [3.835133, -45.577694, 0.406099, 6.475123, -0.387823, -34.038582, -11.949910, -55.761356, 0.749737, 8.917984]"
    );

    // Check first block (token 0) as baseline
    println!("\n=== Checking token 0 ===");
    let block0: &BlockQ6_K =
        unsafe { &*(emb_data.as_ptr() as *const BlockQ6_K) };
    println!(
        "  Block d = 0x{:04x} = {:.6}",
        block0.d,
        half::f16::from_bits(block0.d).to_f32()
    );
    println!("  Block scales[:8] = {:?}", &block0.scales[..8]);

    let mut emb0 = vec![0.0f32; hidden_size];
    for i in 0..nb {
        let block_offset = i * block_size;
        let block_bytes = &emb_data[block_offset..block_offset + block_size];
        dequant_q6_k(
            block_bytes,
            &mut emb0[i * QK_K..],
            QK_K.min(hidden_size - i * QK_K),
        );
    }

    let min0 = emb0.iter().cloned().fold(f32::INFINITY, f32::min);
    let max0 = emb0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean0: f32 = emb0.iter().sum::<f32>() / emb0.len() as f32;

    println!(
        "  Token 0 embedding: min={:.6}, max={:.6}, mean={:.6}",
        min0, max0, mean0
    );
    println!("  First 10: {:?}", &emb0[..10]);
}
