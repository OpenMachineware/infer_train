// Debug F32 weights in Layer 3

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    println!("Loading model: {}", model_path);

    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let model = ctx.model();

    // Check Layer 3 F32 weights
    println!("\n=== Layer 3 F32 Weight Analysis ===");
    if let Some(layer) = model.layers.get(3) {
        // wq
        if let Some(ref wq) = layer.wq {
            println!("wq: dtype={:?}, shape={:?}", wq.dtype, wq.shape);
            println!("  elem_count: {}", wq.elem_count());
            println!("  bytes len: {}", wq.bytes().len());

            if wq.dtype == infer_train::model::tensor::DataType::F32 {
                let bytes = wq.bytes();
                let num_f32 = bytes.len() / 4;
                println!("  Number of F32 values: {}", num_f32);

                // Check the first few F32 values
                let data = infer_train::inference::dequant_to_f32(
                    bytes, wq.dtype, num_f32,
                );
                let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
                let inf_count =
                    data.iter().filter(|&&v| v.is_infinite()).count();
                let max =
                    data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = data.iter().cloned().fold(f32::INFINITY, f32::min);

                println!(
                    "  F32 stats: nan={}, inf={}, range=[{}, {}]",
                    nan_count, inf_count, min, max
                );

                // Check raw bytes
                println!("\n  First 40 bytes of wq:");
                for (i, chunk) in bytes.chunks(4).take(10).enumerate() {
                    let val = f32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ]);
                    println!(
                        "    [{}]: bytes={:02x?} -> f32={}",
                        i, chunk, val
                    );
                }

                // Check if there's a pattern to where NaN appears
                println!("\n  Position of first 10 NaN values:");
                let mut count = 0;
                for (i, &v) in data.iter().enumerate() {
                    if v.is_nan() && count < 10 {
                        let start = i * 4;
                        if start + 4 <= bytes.len() {
                            let raw_bytes = &bytes[start..start + 4];
                            println!(
                                "    pos {}: bytes={:02x?} -> NaN",
                                i, raw_bytes
                            );
                        }
                        count += 1;
                    }
                }
            }
        }

        // wk
        if let Some(ref wk) = layer.wk {
            println!("\nwk: dtype={:?}, shape={:?}", wk.dtype, wk.shape);
            if wk.dtype == infer_train::model::tensor::DataType::F32 {
                let bytes = wk.bytes();
                let num_f32 = bytes.len() / 4;
                let data = infer_train::inference::dequant_to_f32(
                    bytes, wk.dtype, num_f32,
                );
                let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
                let max =
                    data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                println!(
                    "  F32 stats: nan={}, range=[{}, {}]",
                    nan_count, min, max
                );
            }
        }

        // wv (should be Q5_K)
        if let Some(ref wv) = layer.wv {
            println!("\nwv: dtype={:?}, shape={:?}", wv.dtype, wv.shape);
        }
    }
}
