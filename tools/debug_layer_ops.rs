// Debug layer operations to find NaN source
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::{KVCache, KVCacheBase, KVCacheFormat};
use infer_train::model::weights::WeightStore;
use infer_train::model::{GgufFile, ModelConfig, llama::LlamaModel};

fn stats(name: &str, data: &[f32]) {
    let num_nan = data.iter().filter(|&&v| v.is_nan()).count();
    let num_inf = data.iter().filter(|&&v| v.is_infinite()).count();
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean = data.iter().filter(|&&v| v.is_finite()).sum::<f32>()
        / data.iter().filter(|&&v| v.is_finite()).count().max(1) as f32;
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

    // Load model
    let store = WeightStore::from_gguf(model_path).unwrap();
    let gguf = GgufFile::open(model_path).unwrap();
    let raw_config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, raw_config);
    let config = model.config.clone();

    // Print raw GGUF tensor dtype for layer 3
    println!("\n=== GGUF Raw Tensor Info for Layer 3 ===");
    for tensor in &gguf.tensors {
        if tensor.name == "blk.3.ffn_gate.weight" {
            println!(
                "{}: GGUF dtype={:?}, dims={:?}, offset={}",
                tensor.name, tensor.dtype, tensor.dims, tensor.offset
            );

            // Get tensor data
            if let Some(data) = gguf.get_tensor_data(&tensor.name) {
                println!("  data len: {}", data.len());

                // Calculate expected size
                let total_elems: usize =
                    tensor.dims.iter().product::<u64>() as usize;
                let n_blocks = total_elems.div_ceil(256);
                let expected_bytes = n_blocks
                    * std::mem::size_of::<infer_train::quant::types::BlockIQ4XS>(
                    );
                println!(
                    "  expected bytes for IQ4_XS: {} ({} blocks)",
                    expected_bytes, n_blocks
                );
                println!(
                    "  actual/expected ratio: {:.2}x",
                    data.len() as f64 / expected_bytes as f64
                );

                // Check first 8 bytes as hex
                let hex: String =
                    data[0..8].iter().map(|b| format!("{:02x}", b)).collect();
                println!("  first 8 bytes (hex): {}", hex);

                // Decode as IQ4_XS block header
                use infer_train::quant::types::BlockIQ4XS;
                let block: &BlockIQ4XS =
                    unsafe { &*(data.as_ptr() as *const BlockIQ4XS) };
                let d = half::f16::from_bits(block.d).to_f32();
                println!(
                    "  as IQ4_XS block: d={}, scales_h={}, scales_l={:?}",
                    d, block.scales_h, block.scales_l
                );
            }
        }
    }

    // Check if next tensor overlaps
    println!("\n=== Checking tensor overlap ===");
    let mut layer3_tensors: Vec<_> =
        gguf.tensors.iter().filter(|t| t.name.starts_with("blk.3.")).collect();
    layer3_tensors.sort_by_key(|t| t.offset);

    for (i, tensor) in layer3_tensors.iter().enumerate() {
        if let Some(data) = gguf.get_tensor_data(&tensor.name) {
            println!(
                "{}: offset={}, len={}",
                tensor.name,
                tensor.offset,
                data.len()
            );
            if i > 0 {
                let prev = layer3_tensors[i - 1];
                if let Some(prev_data) = gguf.get_tensor_data(&prev.name) {
                    let prev_end = prev.offset as usize + prev_data.len();
                    if tensor.offset < prev_end as u64 {
                        println!("  WARNING: overlaps with previous tensor!");
                    }
                }
            }
        }
    }

    println!("Layers: {}", model.layers.len());
    println!("Hidden size: {}", config.hidden_size);
    println!(
        "n_heads: {}, n_kv_heads: {}, head_dim: {}",
        config.n_heads, config.n_kv_heads, config.head_dim
    );

    // Run through layers until we hit NaN
    let tokens: Vec<u32> = vec![1];
    let hidden_size = config.hidden_size;

    let mut kv = KVCache::new(&config, 16, KVCacheFormat::F32);
    let mut hidden = model.embedding_lookup_impl(tokens[0]).unwrap();
    stats("embedding", &hidden);

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        println!("\n=== Layer {} ===", layer_idx);

        // Print layer weight dtypes
        if let Some(wq) = layer.wq.as_ref() {
            println!("  wq: {:?}, shape={:?}", wq.dtype, wq.shape);
        }
        if let Some(wk) = layer.wk.as_ref() {
            println!("  wk: {:?}, shape={:?}", wk.dtype, wk.shape);
        }
        if let Some(wv) = layer.wv.as_ref() {
            println!("  wv: {:?}, shape={:?}", wv.dtype, wv.shape);
        }
        if let Some(wo) = layer.wo.as_ref() {
            println!("  wo: {:?}, shape={:?}", wo.dtype, wo.shape);
        }
        if let Some(w1) = layer.w1.as_ref() {
            println!("  w1: {:?}, shape={:?}", w1.dtype, w1.shape);
        }
        if let Some(w2) = layer.w2.as_ref() {
            println!("  w2: {:?}, shape={:?}", w2.dtype, w2.shape);
        }
        if let Some(w3) = layer.w3.as_ref() {
            println!("  w3: {:?}, shape={:?}", w3.dtype, w3.shape);
        }

        // 1. Attention norm
        let normed = model
            .apply_rms_norm_impl(
                &hidden,
                layer.attention_norm.as_ref().unwrap(),
                hidden_size,
            )
            .unwrap();
        stats("attn_normed", &normed);

        // Check if norm produced NaN
        if normed.iter().any(|&v| v.is_nan()) {
            println!("  NaN introduced in attention norm!");
            // Check weight
            let w = layer.attention_norm.as_ref().unwrap();
            stats(
                "attn_norm_weight",
                &infer_train::inference::dequant_to_f32(
                    w.bytes(),
                    w.dtype,
                    hidden_size,
                ),
            );
            break;
        }

        // 2. QKV projections
        let wq = layer.wq.as_ref().unwrap();
        let wk = layer.wk.as_ref().unwrap();
        let wv = layer.wv.as_ref().unwrap();

        let q_out_size = wq.shape[1];
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];

        let q = model.gemv_impl(&normed, wq, q_out_size).unwrap();
        stats("q", &q);
        let k = model.gemv_impl(&normed, wk, k_out_size).unwrap();
        stats("k", &k);
        let v = model.gemv_impl(&normed, wv, v_out_size).unwrap();
        stats("v", &v);

        // Check if QKV produced NaN
        if q.iter().any(|&v| v.is_nan()) {
            println!("  NaN in Q projection!");
            break;
        }
        if k.iter().any(|&v| v.is_nan()) {
            println!("  NaN in K projection!");
            break;
        }
        if v.iter().any(|&v| v.is_nan()) {
            println!("  NaN in V projection!");
            break;
        }

        // 3. RoPE
        let actual_n_heads = q_out_size / config.head_dim;
        let actual_n_kv_heads = k_out_size / config.head_dim;
        let pos = kv.seq_len();

        let q_rope =
            model.apply_rope(&q, pos, actual_n_heads, config.head_dim).unwrap();
        let k_rope = model
            .apply_rope(&k, pos, actual_n_kv_heads, config.head_dim)
            .unwrap();
        stats("q_rope", &q_rope);
        stats("k_rope", &k_rope);

        if q_rope.iter().any(|&v| v.is_nan()) {
            println!("  NaN in Q RoPE!");
            break;
        }
        if k_rope.iter().any(|&v| v.is_nan()) {
            println!("  NaN in K RoPE!");
            break;
        }

        // 4. Attention
        let attn_out = model
            .attention(
                &q_rope,
                &k_rope,
                &v,
                layer_idx,
                &mut kv,
                actual_n_heads,
                config.head_dim,
                actual_n_kv_heads,
            )
            .unwrap();
        stats("attn_out", &attn_out);

        if attn_out.iter().any(|&v| v.is_nan()) {
            println!("  NaN in attention!");
            break;
        }

        // 5. Output projection
        let attn_proj = model
            .gemv_impl(&attn_out, layer.wo.as_ref().unwrap(), hidden_size)
            .unwrap();
        stats("attn_proj", &attn_proj);

        if attn_proj.iter().any(|&v| v.is_nan()) {
            println!("  NaN in output projection!");
            break;
        }

        // 6. Residual
        let residual: Vec<f32> =
            hidden.iter().zip(attn_proj.iter()).map(|(h, a)| h + a).collect();
        stats("residual", &residual);

        if residual.iter().any(|&v| v.is_nan()) {
            println!("  NaN in residual!");
            break;
        }

        // 7. FFN norm
        let ffn_normed = model
            .apply_rms_norm_impl(
                &residual,
                layer.ffn_norm.as_ref().unwrap(),
                hidden_size,
            )
            .unwrap();
        stats("ffn_normed", &ffn_normed);

        if ffn_normed.iter().any(|&v| v.is_nan()) {
            println!("  NaN in FFN norm!");
            break;
        }

        // 8. FFN - detailed debugging
        let intermediate = hidden_size * 4;

        // Gate projection
        let w1 = layer.w1.as_ref().unwrap();
        println!("  w1 dtype: {:?}, shape: {:?}", w1.dtype, w1.shape);
        println!("  w1 bytes len: {}", w1.bytes().len());

        // Check if data matches dtype
        let total_elems = w1.shape.iter().product::<usize>();
        let expected_bytes = match w1.dtype {
            infer_train::model::tensor::DataType::F32 => total_elems * 4,
            infer_train::model::tensor::DataType::F16 => total_elems * 2,
            infer_train::model::tensor::DataType::IQ4_XS => {
                let n_blocks = total_elems.div_ceil(256);
                n_blocks
                    * std::mem::size_of::<infer_train::quant::types::BlockIQ4XS>(
                    )
            }
            _ => total_elems, // default assumption
        };
        println!("  Expected bytes for dtype: {}", expected_bytes);

        if w1.bytes().len() != expected_bytes {
            println!(
                "  WARNING: byte size mismatch! Actual {} != expected {}",
                w1.bytes().len(),
                expected_bytes
            );

            // Try to interpret as F32
            if w1.bytes().len() == total_elems * 4 {
                println!("  Data size matches F32 ({} bytes)", total_elems * 4);
                // Read first few F32 values
                let f32_ptr = w1.bytes().as_ptr() as *const f32;
                let first_vals: Vec<f32> =
                    unsafe { (0..4).map(|i| *f32_ptr.add(i)).collect() };
                println!("  First 4 F32 values: {:?}", first_vals);
            }
        }

        // Debug: check first few blocks of IQ4_XS weight
        if matches!(w1.dtype, infer_train::model::tensor::DataType::IQ4_XS) {
            use infer_train::quant::types::BlockIQ4XS;
            const QK_K: usize = 256;
            const BLOCK_SIZE: usize = std::mem::size_of::<BlockIQ4XS>();

            let expected_blocks = total_elems / QK_K;

            let block: &BlockIQ4XS =
                unsafe { &*(w1.bytes().as_ptr() as *const BlockIQ4XS) };
            let d = half::f16::from_bits(block.d).to_f32();
            println!("  First block d: {}", d);
            println!(
                "  First block scales_h: {:016b} ({} decimal)",
                block.scales_h, block.scales_h
            );
            println!("  First block scales_l: {:?}", block.scales_l);

            // Calculate ls for first sub-block
            let ls0 = ((block.scales_l[0] >> 0) & 0xF) as i32
                | ((((block.scales_h >> 0) & 3) << 4) as i32);
            let ls1 = ((block.scales_l[0] >> 4) & 0xF) as i32
                | ((((block.scales_h >> 2) & 3) << 4) as i32);
            println!("  ls[0] = {}, dl[0] = {}", ls0, d * (ls0 - 32) as f32);
            println!("  ls[1] = {}, dl[1] = {}", ls1, d * (ls1 - 32) as f32);

            // Check first 4 values
            let kvalues: [f32; 16] = [
                -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0,
                13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
            ];
            let dl = d * (ls0 - 32) as f32;
            for j in 0..4 {
                let byte = block.qs[j];
                let idx0 = (byte & 0x0F) as usize;
                let idx1 = ((byte >> 4) & 0x0F) as usize;
                let v0 = kvalues[idx0] * dl;
                let v1 = kvalues[idx1] * dl;
                println!("  qs[{}] = {:08b}, values: {}, {}", j, byte, v0, v1);
            }
        }

        let gate = model.gemv(&ffn_normed, w1, intermediate).unwrap();
        stats("gate", &gate);

        // Up projection
        let w3 = layer.w3.as_ref().unwrap();
        println!("  w3 dtype: {:?}, shape: {:?}", w3.dtype, w3.shape);
        let up = model.gemv(&ffn_normed, w3, intermediate).unwrap();
        stats("up", &up);

        // SiLU activation
        let silu: Vec<f32> =
            gate.iter().map(|g| g / (1.0 + (-g).exp())).collect();
        stats("silu", &silu);

        // Element-wise multiply
        let hidden_ffn: Vec<f32> =
            silu.iter().zip(up.iter()).map(|(s, u)| s * u).collect();
        stats("hidden_ffn", &hidden_ffn);

        if hidden_ffn.iter().any(|&v| v.is_nan()) {
            println!("  NaN in SiLU * up!");
            // Check for extreme values
            let _gate_max =
                gate.iter().cloned().fold(f32::NEG_INFINITY, f32::max).abs();
            let silu_extreme = gate.iter().filter(|&&g| g.abs() > 20.0).count();
            println!("  gate extreme values (|g| > 20): {}", silu_extreme);
            break;
        }

        // Down projection
        let w2 = layer.w2.as_ref().unwrap();
        println!("  w2 dtype: {:?}, shape: {:?}", w2.dtype, w2.shape);
        let ffn_out = model.gemv_impl(&hidden_ffn, w2, hidden_size).unwrap();
        stats("ffn_out", &ffn_out);

        if ffn_out.iter().any(|&v| v.is_nan()) {
            println!("  NaN in FFN down projection!");

            // Check if the issue is in dequantization
            let w2_data = infer_train::inference::dequant_to_f32(
                w2.bytes(),
                w2.dtype,
                intermediate * hidden_size,
            );
            let w2_nan = w2_data.iter().filter(|&&v| v.is_nan()).count();
            println!("  w2 dequant NaN count: {}/{}", w2_nan, w2_data.len());

            // Check hidden input range
            let hidden_max =
                hidden_ffn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let hidden_min =
                hidden_ffn.iter().cloned().fold(f32::INFINITY, f32::min);
            println!("  hidden range: [{}, {}]", hidden_min, hidden_max);
            break;
        }

        // 9. Final residual
        hidden =
            residual.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect();
        stats("layer_out", &hidden);

        if hidden.iter().any(|&v| v.is_nan()) {
            println!("  NaN in final residual!");
            break;
        }

        // Note: Don't advance KV cache here - it should only advance once per token
    }

    // Advance KV cache once after processing all layers
    kv.advance();
}
