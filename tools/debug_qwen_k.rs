// Debug Qwen3-0.6B K projection all-zeros
use infer_train::inference::{dequant_to_f32, Device, InferenceContext};
use infer_train::model::tensor::DataType;

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

    // Check wk weight details
    if let Some(ref wk) = layer.wk {
        println!("wk weight:");
        println!("  shape: {:?}", wk.shape);
        println!("  dtype: {:?}", wk.dtype);
        println!("  bytes len: {}", wk.bytes().len());

        // Try to dequantize first few elements
        if wk.dtype == DataType::Q5_K {
            let total_elems = wk.elem_count();
            println!("  total elems: {}", total_elems);

            // Dequantize and check first few values
            let dequant = dequant_to_f32(wk.bytes(), wk.dtype, total_elems.min(256));
            println!("  first 10 dequant values: {:?}",
                &dequant[..10.min(dequant.len())]);

            // Check for all zeros
            let zero_count = dequant.iter().filter(|&&x| x == 0.0).count();
            println!("  zeros in first 256: {}/256", zero_count);
        }
    } else {
        println!("No wk weight!");
    }

    // Also check wq and wv for comparison
    if let Some(ref wq) = layer.wq {
        println!("\nwq weight:");
        println!("  shape: {:?}", wq.shape);
        println!("  dtype: {:?}", wq.dtype);
    }

    if let Some(ref wv) = layer.wv {
        println!("\nwv weight:");
        println!("  shape: {:?}", wv.shape);
        println!("  dtype: {:?}", wv.dtype);
    }

    // Test GEMV with a simple input
    let test_token = 1u32;
    let hidden = ctx.model.embedding_lookup_impl(test_token).unwrap();
    let hidden_size = ctx.config.hidden_size;

    // Attention norm
    let attn_norm = layer.attention_norm.as_ref().unwrap();
    let normed = ctx.model.apply_rms_norm_impl(&hidden, attn_norm, hidden_size).unwrap();

    println!("\nTesting GEMV:");
    println!("  normed (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        normed[0], normed[1], normed[2], normed[3], normed[4]);

    // Test Q projection
    if let Some(ref wq) = layer.wq {
        let q_size = wq.shape[1];
        let q = ctx.model.gemv_impl(&normed, wq, q_size).unwrap();
        println!("  Q (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
            q[0], q[1], q[2], q[3], q[4]);
    }

    // Test K projection
    if let Some(ref wk) = layer.wk {
        let k_size = wk.shape[1];
        println!("\n  K projection size: {}", k_size);
        let k = ctx.model.gemv_impl(&normed, wk, k_size);
        match k {
            Some(k_vec) => {
                let zero_count = k_vec.iter().filter(|&&x| x == 0.0).count();
                println!("  K result len: {}", k_vec.len());
                println!("  K zeros: {}/{}", zero_count, k_vec.len());
                println!("  K (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
                    k_vec[0], k_vec[1], k_vec[2], k_vec[3], k_vec[4]);
            }
            None => {
                println!("  K GEMV returned None!");
            }
        }
    }

    // Test V projection
    if let Some(ref wv) = layer.wv {
        let v_size = wv.shape[1];
        let v = ctx.model.gemv_impl(&normed, wv, v_size).unwrap();
        println!("  V (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
            v[0], v[1], v[2], v[3], v[4]);
    }
}
