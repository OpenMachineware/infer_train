// Debug layer-by-layer comparison between CPU and GPU
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::KVCache;
use infer_train::inference::KVCacheFormat;
use infer_train::model::unified::{Architecture, Model};
use infer_train::model::weights::WeightStore;
use infer_train::model::{GgufFile, ModelConfig, llama::LlamaModel};

fn stats(name: &str, data: &[f32]) {
    let num_nan = data.iter().filter(|&&v| v.is_nan()).count();
    let num_inf = data.iter().filter(|&&v| v.is_infinite()).count();
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    println!(
        "  {}: len={}, nan={}, inf={}, range=[{:.4}, {:.4}], mean={:.4}",
        name,
        data.len(),
        num_nan,
        num_inf,
        min,
        max,
        mean
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    // Load unified model with architecture detection
    let unified_model =
        Model::from_gguf(model_path).expect("Failed to load model");
    println!("Architecture: {:?}", unified_model.architecture);

    let config = unified_model.config.clone();

    println!("Layers: {}", unified_model.layer_count());
    println!("Hidden size: {}", config.hidden_size);
    println!(
        "n_heads: {}, n_kv_heads: {}, head_dim: {}",
        config.n_heads, config.n_kv_heads, config.head_dim
    );

    // Test tokens
    let tokens: Vec<u32> = vec![1];
    let hidden_size = config.hidden_size;

    // Get embedding
    let hidden = if let Some(embd) = &unified_model.tok_embeddings {
        let embd_data = infer_train::inference::dequant_to_f32(
            embd.bytes(),
            embd.dtype,
            embd.shape[0] * embd.shape[1],
        );
        let vocab_size = embd.shape[1];
        let offset = tokens[0] as usize * hidden_size;
        if offset + hidden_size <= embd_data.len() {
            embd_data[offset..offset + hidden_size].to_vec()
        } else {
            vec![0.0f32; hidden_size]
        }
    } else {
        vec![0.0f32; hidden_size]
    };

    stats("embedding", &hidden);

    // Forward pass based on architecture
    match unified_model.architecture {
        Architecture::Llama | Architecture::MoE => {
            // Use LlamaModel for these architectures
            let store = WeightStore::from_gguf(model_path).unwrap();
            let gguf = GgufFile::open(model_path).unwrap();
            let raw_config = ModelConfig::from_gguf(&gguf);
            let model = LlamaModel::from_weights(&store, raw_config);
            let config = model.config.clone();

            println!("\n=== CPU Forward (Llama/MoE) ===");

            // Print weight dtypes
            if let Some(layer) = model.layers.first() {
                println!("\nLayer 0 weights:");
                if let Some(wq) = layer.wq.as_ref() {
                    println!("  wq: {:?}, shape={:?}", wq.dtype, wq.shape);
                }
                if let Some(wk) = layer.wk.as_ref() {
                    println!("  wk: {:?}, shape={:?}", wk.dtype, wk.shape);
                }
                if let Some(wv) = layer.wv.as_ref() {
                    println!("  wv: {:?}, shape={:?}", wv.dtype, wv.shape);
                }
            }

            let mut cpu_kv = KVCache::new(&config, 16, KVCacheFormat::F32);
            let mut cpu_hidden = hidden.clone();

            for (layer_idx, layer) in model.layers.iter().enumerate() {
                // For layer 1, print MoE-specific info
                if layer_idx == 1 {
                    println!("\n=== Layer 1 MoE Debug ===");
                    println!("Has MoE: {}", layer.ffn_gate_exps.is_some());
                    if let Some(gate_inp) = &layer.ffn_gate_inp {
                        println!("gate_inp shape: {:?}", gate_inp.shape);
                        let weight: &[f32] =
                            bytemuck::cast_slice(gate_inp.bytes());
                        println!(
                            "gate_inp first 10 weights per expert 0: {:?}",
                            &weight[0..10]
                        );
                    }
                    if let Some(gate_exps) = &layer.ffn_gate_exps {
                        println!(
                            "gate_exps shape: {:?}, dtype: {:?}",
                            gate_exps.shape, gate_exps.dtype
                        );
                        println!(
                            "gate_exps bytes len: {}",
                            gate_exps.bytes().len()
                        );
                    }
                    if let Some(down_exps) = &layer.ffn_down_exps {
                        println!(
                            "down_exps shape: {:?}, dtype: {:?}",
                            down_exps.shape, down_exps.dtype
                        );
                    }
                    println!("n_experts_used: {}", config.n_experts_used);
                }

                cpu_hidden = model
                    .forward_layer(
                        layer,
                        layer_idx,
                        &cpu_hidden,
                        &mut cpu_kv,
                        config.n_heads,
                        config.head_dim,
                        config.n_kv_heads,
                    )
                    .expect(&format!("Layer {} forward failed", layer_idx));

                stats(&format!("layer {} out", layer_idx), &cpu_hidden);
            }
        }
        Architecture::Mamba | Architecture::Hybrid => {
            println!("\n=== CPU Forward (SSM/Hybrid) ===");

            // Print layer 0 weights for Mamba
            if unified_model.architecture == Architecture::Mamba
                && !unified_model.mamba_layers.is_empty()
            {
                let layer = &unified_model.mamba_layers[0];
                println!("\nLayer 0 weights:");
                if let Some(attn_qkv) = &layer.attn_qkv {
                    println!(
                        "  attn_qkv: {:?}, shape={:?}",
                        attn_qkv.dtype, attn_qkv.shape
                    );
                }
                if let Some(attn_gate) = &layer.attn_gate {
                    println!(
                        "  attn_gate: {:?}, shape={:?}",
                        attn_gate.dtype, attn_gate.shape
                    );
                }
                if let Some(ssm_out) = &layer.ssm_out {
                    println!(
                        "  ssm_out: {:?}, shape={:?}",
                        ssm_out.dtype, ssm_out.shape
                    );
                }
            }

            let mut cpu_kv = KVCache::new(&config, 16, KVCacheFormat::F32);
            let mut cpu_hidden = hidden.clone();

            for layer_idx in 0..unified_model.layer_count() {
                cpu_hidden = unified_model
                    .forward_layer(&cpu_hidden, layer_idx, &mut cpu_kv)
                    .expect(&format!("Layer {} forward failed", layer_idx));

                stats(&format!("layer {} out", layer_idx), &cpu_hidden);
            }
        }
        Architecture::Unknown => {
            println!("Unknown architecture");
        }
    }
}
