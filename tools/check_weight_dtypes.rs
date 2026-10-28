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

    println!("=== Weight Data Types ===\n");

    // Check embedding dtype
    if let Some(emb) = &ctx.model.tok_embeddings {
        println!("Embedding: {:?}", emb.dtype);
    }

    // Check layer 0 weights
    if let Some(layer) = ctx.model.layers.first() {
        println!("\nLayer 0:");
        if let Some(wq) = &layer.wq {
            println!("  wq: {:?} shape: {:?}", wq.dtype, wq.shape);
        }
        if let Some(wk) = &layer.wk {
            println!("  wk: {:?} shape: {:?}", wk.dtype, wk.shape);
        }
        if let Some(wv) = &layer.wv {
            println!("  wv: {:?} shape: {:?}", wv.dtype, wv.shape);
        }
        if let Some(wo) = &layer.wo {
            println!("  wo: {:?} shape: {:?}", wo.dtype, wo.shape);
        }
        if let Some(w1) = &layer.w1 {
            println!("  w1 (gate): {:?} shape: {:?}", w1.dtype, w1.shape);
        }
        if let Some(w2) = &layer.w2 {
            println!("  w2 (down): {:?} shape: {:?}", w2.dtype, w2.shape);
        }
        if let Some(w3) = &layer.w3 {
            println!("  w3 (up): {:?} shape: {:?}", w3.dtype, w3.shape);
        }
    }

    // Check output layer
    if let Some(output) = &ctx.model.output {
        println!(
            "\nOutput layer: {:?} shape: {:?}",
            output.dtype, output.shape
        );
    }
}
