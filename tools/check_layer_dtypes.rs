// SPDX-License-Identifier: AGPL-3.0-or-later
// Check layer dtypes in a GGUF model

use infer_train::model::weights::WeightStore;
use infer_train::model::{GgufFile, LlamaModel, ModelConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("Loading model from {}...", model_path);

    let store =
        WeightStore::from_gguf(model_path).expect("Failed to load weights");
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let raw_config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, raw_config);

    println!("\nModel has {} layers", model.layers.len());

    // Check first layer
    if let Some(layer) = model.layers.first() {
        println!("\nLayer 0 tensor dtypes:");
        if let Some(wq) = &layer.wq {
            println!("  wq: {:?} shape {:?}", wq.dtype, wq.shape);
        }
        if let Some(wk) = &layer.wk {
            println!("  wk: {:?} shape {:?}", wk.dtype, wk.shape);
        }
        if let Some(wv) = &layer.wv {
            println!("  wv: {:?} shape {:?}", wv.dtype, wv.shape);
        }
        if let Some(wo) = &layer.wo {
            println!("  wo: {:?} shape {:?}", wo.dtype, wo.shape);
        }
        if let Some(w1) = &layer.w1 {
            println!("  w1 (FFN gate): {:?} shape {:?}", w1.dtype, w1.shape);
        }
        if let Some(w2) = &layer.w2 {
            println!("  w2 (FFN down): {:?} shape {:?}", w2.dtype, w2.shape);
        }
        if let Some(w3) = &layer.w3 {
            println!("  w3 (FFN up): {:?} shape {:?}", w3.dtype, w3.shape);
        }
        if let Some(attn_qkv) = &layer.attn_qkv {
            println!(
                "  attn_qkv: {:?} shape {:?}",
                attn_qkv.dtype, attn_qkv.shape
            );
        }
    }

    // Check output layer
    if let Some(output) = &model.output {
        println!(
            "\nOutput projection: {:?} shape {:?}",
            output.dtype, output.shape
        );
    }
}
