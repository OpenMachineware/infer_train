// Test first layer correctness
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::InferenceContext;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string()
    });

    println!("=== First Layer Test ===\n");

    let gpu_ctx = MetalContext::new().unwrap();
    let device = gpu_ctx.device().clone();
    let ctx =
        InferenceContext::from_gguf_with_gpu_ctx(&model_path, gpu_ctx.clone())
            .unwrap();

    println!("Model loaded: {} layers", ctx.model.layers.len());
    println!("Hidden size: {}", ctx.model.config.hidden_size);

    // Test embedding lookup
    let token = 9707u32; // Same as nan_source test
    let hidden = ctx.model.embedding_lookup_impl(token).unwrap();

    println!("\n=== Embedding Test ===");
    println!("Token: {}", token);
    println!("Hidden size: {}", hidden.len());
    println!("First 10 values: {:?}", &hidden[..10]);
    println!("Has NaN: {}", hidden.iter().any(|&x| x.is_nan()));

    // Check min/max
    let min = hidden.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("Min: {}, Max: {}", min, max);

    // Upload to GPU and download back
    let hidden_buffer = device.new_buffer_with_data(
        hidden.as_ptr() as *const std::ffi::c_void,
        (hidden.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let mut downloaded = vec![0.0f32; hidden.len()];
    unsafe {
        std::ptr::copy_nonoverlapping(
            hidden_buffer.contents() as *const f32,
            downloaded.as_mut_ptr(),
            hidden.len(),
        );
    }

    println!("\nAfter GPU roundtrip:");
    println!("First 10 values: {:?}", &downloaded[..10]);
    println!("Has NaN: {}", downloaded.iter().any(|&x| x.is_nan()));

    // Test first layer RMS Norm
    println!("\n=== RMS Norm Test ===");
    let layer = &ctx.model.layers[0];
    let norm_weight = layer.attention_norm.as_ref().unwrap();
    println!("Norm weight shape: {:?}", norm_weight.shape);
    println!("Norm weight dtype: {:?}", norm_weight.dtype);

    // Create command buffer for single operation
    let queue = gpu_ctx.queue();
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Allocate buffers
    let hidden_size = ctx.model.config.hidden_size;
    let input_buf = std::rc::Rc::new(hidden_buffer);
    let weight_buf = gpu_ctx.get_or_create_buffer(norm_weight);
    let output_buf = std::rc::Rc::new(device.new_buffer(
        (hidden_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    // Encode RMS norm
    gpu_ctx.rms_norm_encode(
        encoder,
        &input_buf,
        &weight_buf,
        &output_buf,
        1,
        hidden_size,
        1e-5,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Download result
    let mut normed = vec![0.0f32; hidden_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            output_buf.contents() as *const f32,
            normed.as_mut_ptr(),
            hidden_size,
        );
    }

    println!("Normed output first 10: {:?}", &normed[..10]);
    println!("Has NaN: {}", normed.iter().any(|&x| x.is_nan()));

    // Check if norm weight is correct
    println!("\nNorm weight info:");
    println!("  dtype: {:?}", norm_weight.dtype);
    println!("  shape: {:?}", norm_weight.shape);
    println!("  size: {} bytes", norm_weight.bytes().len());

    // Test complete layer 0 forward with per-operation checking
    println!("\n=== Complete Layer 0 Forward ===");
    let layer = &ctx.model.layers[0];

    let wq = layer.wq.as_ref().unwrap();
    let wk = layer.wk.as_ref().unwrap();
    let wv = layer.wv.as_ref().unwrap();
    let wo = layer.wo.as_ref().unwrap();

    let q_out_size = wq.shape[1];
    let k_out_size = wk.shape[1];
    let v_out_size = wv.shape[1];

    // Create all buffers
    let q_buf = std::rc::Rc::new(device.new_buffer(
        (q_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));
    let k_buf = std::rc::Rc::new(device.new_buffer(
        (k_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));
    let v_buf = std::rc::Rc::new(device.new_buffer(
        (v_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let wq_buf = gpu_ctx.get_or_create_buffer(wq);
    let wk_buf = gpu_ctx.get_or_create_buffer(wk);
    let wv_buf = gpu_ctx.get_or_create_buffer(wv);

    // Test: single CB with all 3 GEMV operations
    println!("\n--- Single CB with 3 GEMV operations ---");
    let command_buffer3 = queue.new_command_buffer();
    let encoder3 = command_buffer3.new_compute_command_encoder();

    // Encode all 3 projections in one CB
    gpu_ctx.mv_encode(
        encoder3,
        &wq_buf,
        &output_buf,
        &q_buf,
        q_out_size,
        hidden_size,
        wq.dtype,
    );
    gpu_ctx.mv_encode(
        encoder3,
        &wk_buf,
        &output_buf,
        &k_buf,
        k_out_size,
        hidden_size,
        wk.dtype,
    );
    gpu_ctx.mv_encode(
        encoder3,
        &wv_buf,
        &output_buf,
        &v_buf,
        v_out_size,
        hidden_size,
        wv.dtype,
    );

    encoder3.end_encoding();
    command_buffer3.commit();
    command_buffer3.wait_until_completed();

    // Check all results
    let mut q_vals = vec![0.0f32; q_out_size];
    let mut k_vals = vec![0.0f32; k_out_size];
    let mut v_vals = vec![0.0f32; v_out_size];

    unsafe {
        std::ptr::copy_nonoverlapping(
            q_buf.contents() as *const f32,
            q_vals.as_mut_ptr(),
            q_out_size,
        );
        std::ptr::copy_nonoverlapping(
            k_buf.contents() as *const f32,
            k_vals.as_mut_ptr(),
            k_out_size,
        );
        std::ptr::copy_nonoverlapping(
            v_buf.contents() as *const f32,
            v_vals.as_mut_ptr(),
            v_out_size,
        );
    }

    let q_nan = q_vals.iter().filter(|&&x| x.is_nan()).count();
    let k_nan = k_vals.iter().filter(|&&x| x.is_nan()).count();
    let v_nan = v_vals.iter().filter(|&&x| x.is_nan()).count();

    println!("Q: {} NaN / {}", q_nan, q_out_size);
    println!("K: {} NaN / {}", k_nan, k_out_size);
    println!("V: {} NaN / {}", v_nan, v_out_size);

    // Test: separate CB for each operation
    println!("\n--- Separate CB for each GEMV ---");

    // Q
    let q_buf2 = std::rc::Rc::new(device.new_buffer(
        (q_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));
    let cb_q = queue.new_command_buffer();
    let enc_q = cb_q.new_compute_command_encoder();
    gpu_ctx.mv_encode(
        enc_q,
        &wq_buf,
        &output_buf,
        &q_buf2,
        q_out_size,
        hidden_size,
        wq.dtype,
    );
    enc_q.end_encoding();
    cb_q.commit();
    cb_q.wait_until_completed();

    let mut q_vals2 = vec![0.0f32; q_out_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            q_buf2.contents() as *const f32,
            q_vals2.as_mut_ptr(),
            q_out_size,
        );
    }
    let q_nan2 = q_vals2.iter().filter(|&&x| x.is_nan()).count();
    println!("Q (separate CB): {} NaN / {}", q_nan2, q_out_size);

    // K
    let k_buf2 = std::rc::Rc::new(device.new_buffer(
        (k_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));
    let cb_k = queue.new_command_buffer();
    let enc_k = cb_k.new_compute_command_encoder();
    gpu_ctx.mv_encode(
        enc_k,
        &wk_buf,
        &output_buf,
        &k_buf2,
        k_out_size,
        hidden_size,
        wk.dtype,
    );
    enc_k.end_encoding();
    cb_k.commit();
    cb_k.wait_until_completed();

    let mut k_vals2 = vec![0.0f32; k_out_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            k_buf2.contents() as *const f32,
            k_vals2.as_mut_ptr(),
            k_out_size,
        );
    }
    let k_nan2 = k_vals2.iter().filter(|&&x| x.is_nan()).count();
    println!("K (separate CB): {} NaN / {}", k_nan2, k_out_size);

    // V
    let v_buf2 = std::rc::Rc::new(device.new_buffer(
        (v_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));
    let cb_v = queue.new_command_buffer();
    let enc_v = cb_v.new_compute_command_encoder();
    gpu_ctx.mv_encode(
        enc_v,
        &wv_buf,
        &output_buf,
        &v_buf2,
        v_out_size,
        hidden_size,
        wv.dtype,
    );
    enc_v.end_encoding();
    cb_v.commit();
    cb_v.wait_until_completed();

    let mut v_vals2 = vec![0.0f32; v_out_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            v_buf2.contents() as *const f32,
            v_vals2.as_mut_ptr(),
            v_out_size,
        );
    }
    let v_nan2 = v_vals2.iter().filter(|&&x| x.is_nan()).count();
    println!("V (separate CB): {} NaN / {}", v_nan2, v_out_size);

    // Test the full layer forward pass using the model's forward_layer method
    println!("\n=== Testing Full Layer 0 Forward via Model ===");

    // Get config values
    let n_heads = ctx.model.config.n_heads;
    let n_kv_heads = ctx.model.config.n_kv_heads;
    let head_dim = k_out_size / n_kv_heads;

    println!(
        "n_heads: {}, n_kv_heads: {}, head_dim: {}",
        n_heads, n_kv_heads, head_dim
    );

    // Create KV cache for layer 0
    let kv_cache_size = n_kv_heads * 256 * head_dim * 2; // 256 positions max
    let kv_cache_k = std::rc::Rc::new(device.new_buffer(
        kv_cache_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));
    let kv_cache_v = std::rc::Rc::new(device.new_buffer(
        kv_cache_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    // Zero KV cache
    unsafe {
        std::ptr::write_bytes(
            kv_cache_k.contents() as *mut u8,
            0,
            kv_cache_size,
        );
        std::ptr::write_bytes(
            kv_cache_v.contents() as *mut u8,
            0,
            kv_cache_size,
        );
    }

    // Copy K and V to cache position 0 (as F16)
    // K and V are F32, need to convert to F16 for KV cache
    let k_f16: Vec<half::f16> =
        k_vals2.iter().map(|&x| half::f16::from_f32(x)).collect();
    let v_f16: Vec<half::f16> =
        v_vals2.iter().map(|&x| half::f16::from_f32(x)).collect();

    unsafe {
        std::ptr::copy_nonoverlapping(
            k_f16.as_ptr() as *const u8,
            kv_cache_k.contents() as *mut u8,
            k_f16.len() * 2,
        );
        std::ptr::copy_nonoverlapping(
            v_f16.as_ptr() as *const u8,
            kv_cache_v.contents() as *mut u8,
            v_f16.len() * 2,
        );
    }

    // Run Flash Attention MMA (only for head_dim=128)
    if head_dim == 128 {
        println!("\n--- Testing Flash Attention MMA ---");
        let attn_out = std::rc::Rc::new(device.new_buffer(
            (q_out_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let cb_attn = queue.new_command_buffer();
        let enc_attn = cb_attn.new_compute_command_encoder();

        gpu_ctx.flash_attn_mma_encode(
            enc_attn,
            n_heads,
            n_kv_heads,
            1,   // seq_len
            256, // kv_stride
            &q_buf2,
            &kv_cache_k,
            &kv_cache_v,
            &attn_out,
        );

        enc_attn.end_encoding();
        cb_attn.commit();
        cb_attn.wait_until_completed();

        let mut attn_vals = vec![0.0f32; q_out_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                attn_out.contents() as *const f32,
                attn_vals.as_mut_ptr(),
                q_out_size,
            );
        }
        let attn_nan = attn_vals.iter().filter(|&&x| x.is_nan()).count();
        println!("Attention output NaN: {} / {}", attn_nan, q_out_size);
        if attn_nan > 0 {
            for i in 0..std::cmp::min(20, q_out_size) {
                if attn_vals[i].is_nan() {
                    println!("  [{}] = NaN", i);
                }
            }
        } else {
            println!("First 10 attn values: {:?}", &attn_vals[..10]);
        }
    } else {
        println!("head_dim = {}, skipping MMA kernel test", head_dim);
    }

    // Test output projection (wo) with Q directly (skip attention for now)
    println!("\n--- Testing Output Projection (wo) ---");
    let wo = layer.wo.as_ref().unwrap();
    let wo_buf = gpu_ctx.get_or_create_buffer(wo);
    let wo_out_size = wo.shape[1];
    let wo_out = std::rc::Rc::new(device.new_buffer(
        (wo_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let cb_wo = queue.new_command_buffer();
    let enc_wo = cb_wo.new_compute_command_encoder();
    gpu_ctx.mv_encode(
        enc_wo,
        &wo_buf,
        &q_buf2,
        &wo_out,
        wo_out_size,
        q_out_size,
        wo.dtype,
    );
    enc_wo.end_encoding();
    cb_wo.commit();
    cb_wo.wait_until_completed();

    let mut wo_vals = vec![0.0f32; wo_out_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            wo_out.contents() as *const f32,
            wo_vals.as_mut_ptr(),
            wo_out_size,
        );
    }
    let wo_nan = wo_vals.iter().filter(|&&x| x.is_nan()).count();
    println!("WO output NaN: {} / {}", wo_nan, wo_out_size);
    println!("First 5 WO values: {:?}", &wo_vals[..5]);

    // Test residual add
    println!("\n--- Testing Residual Add ---");
    let residual_out = std::rc::Rc::new(device.new_buffer(
        (wo_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let cb_add = queue.new_command_buffer();
    let enc_add = cb_add.new_compute_command_encoder();
    gpu_ctx.add_encode(
        enc_add,
        &output_buf,
        &wo_out,
        &residual_out,
        wo_out_size,
    );
    enc_add.end_encoding();
    cb_add.commit();
    cb_add.wait_until_completed();

    let mut residual_vals = vec![0.0f32; wo_out_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            residual_out.contents() as *const f32,
            residual_vals.as_mut_ptr(),
            wo_out_size,
        );
    }
    let residual_nan = residual_vals.iter().filter(|&&x| x.is_nan()).count();
    println!("Residual output NaN: {} / {}", residual_nan, wo_out_size);
    println!("First 5 residual values: {:?}", &residual_vals[..5]);

    // Test FFN
    println!("\n--- Testing FFN ---");
    let ffn_norm = layer.ffn_norm.as_ref().unwrap();
    let ffn_norm_buf = gpu_ctx.get_or_create_buffer(ffn_norm);

    // FFN norm
    let ffn_normed = std::rc::Rc::new(device.new_buffer(
        (wo_out_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let cb_fn = queue.new_command_buffer();
    let enc_fn = cb_fn.new_compute_command_encoder();
    gpu_ctx.rms_norm_encode(
        enc_fn,
        &residual_out,
        &ffn_norm_buf,
        &ffn_normed,
        1,
        wo_out_size,
        1e-5,
    );
    enc_fn.end_encoding();
    cb_fn.commit();
    cb_fn.wait_until_completed();

    let mut ffn_normed_vals = vec![0.0f32; wo_out_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            ffn_normed.contents() as *const f32,
            ffn_normed_vals.as_mut_ptr(),
            wo_out_size,
        );
    }
    let ffn_norm_nan = ffn_normed_vals.iter().filter(|&&x| x.is_nan()).count();
    println!("FFN norm output NaN: {} / {}", ffn_norm_nan, wo_out_size);

    // Gate + Up projections
    let w1 = layer.w1.as_ref().unwrap();
    let w3 = layer.w3.as_ref().unwrap();
    let intermediate_size = w1.shape[1];

    let w1_buf = gpu_ctx.get_or_create_buffer(w1);
    let w3_buf = gpu_ctx.get_or_create_buffer(w3);

    let gate_out = std::rc::Rc::new(device.new_buffer(
        (intermediate_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));
    let up_out = std::rc::Rc::new(device.new_buffer(
        (intermediate_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    // Single CB for both
    let cb_gate = queue.new_command_buffer();
    let enc_gate = cb_gate.new_compute_command_encoder();
    gpu_ctx.mv_encode(
        enc_gate,
        &w1_buf,
        &ffn_normed,
        &gate_out,
        intermediate_size,
        wo_out_size,
        w1.dtype,
    );
    gpu_ctx.mv_encode(
        enc_gate,
        &w3_buf,
        &ffn_normed,
        &up_out,
        intermediate_size,
        wo_out_size,
        w3.dtype,
    );
    enc_gate.end_encoding();
    cb_gate.commit();
    cb_gate.wait_until_completed();

    let mut gate_vals = vec![0.0f32; intermediate_size];
    let mut up_vals = vec![0.0f32; intermediate_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            gate_out.contents() as *const f32,
            gate_vals.as_mut_ptr(),
            intermediate_size,
        );
        std::ptr::copy_nonoverlapping(
            up_out.contents() as *const f32,
            up_vals.as_mut_ptr(),
            intermediate_size,
        );
    }
    let gate_nan = gate_vals.iter().filter(|&&x| x.is_nan()).count();
    let up_nan = up_vals.iter().filter(|&&x| x.is_nan()).count();
    println!("Gate output NaN: {} / {}", gate_nan, intermediate_size);
    println!("Up output NaN: {} / {}", up_nan, intermediate_size);
}
