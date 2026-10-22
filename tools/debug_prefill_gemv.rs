// Debug prefill to check if GEMV works
use infer_train::inference::InferenceContext;
use std::rc::Rc;

fn main() {
    let model_path = std::env::args().nth(1).expect("Need model path");

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    ).expect("Failed to load model");

    println!("Model loaded: {} layers", ctx.model.layers.len());
    println!("Hidden size: {}", ctx.model.config.hidden_size);

    // Check first layer weight
    if let Some(layer) = ctx.model.layers.first() {
        if let Some(wq) = &layer.wq {
            println!("\nwq weight:");
            println!("  shape: {:?}", wq.shape);
            println!("  dtype: {:?}", wq.dtype);
            println!("  block_type: {:?}", wq.block_type);
            println!("  data len: {} bytes", wq.data.len());

            // Check for reasonable scale values
            if let Some(block_type) = wq.block_type {
                match block_type.as_str() {
                    "q5_k" | "Q5_K" => {
                        use infer_train::quant::types::BlockQ5K;
                        let num_blocks = wq.data.len() / std::mem::size_of::<BlockQ5K>();
                        println!("  num_blocks: {}", num_blocks);

                        let blocks: &[BlockQ5K] = unsafe {
                            std::slice::from_raw_parts(
                                wq.data.as_ptr() as *const BlockQ5K,
                                num_blocks
                            )
                        };

                        // Check first few blocks
                        for (i, block) in blocks.iter().take(5).enumerate() {
                            let d = half::f16::from_bits(block.d).to_f32();
                            let dmin = half::f16::from_bits(block.dmin).to_f32();
                            println!("  Block {}: d={:.4}, dmin={:.4}", i, d, dmin);
                        }
                    }
                    _ => println!("  Block type: {}", block_type)
                }
            }
        }
    }

    // Try a single GEMV operation
    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU");

    // Create a simple input vector
    let hidden_size = ctx.model.config.hidden_size;
    let input: Vec<f32> = (0..hidden_size).map(|i| (i % 100) as f32 / 100.0).collect();
    let input_buffer = Rc::new(gpu_ctx.device().new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (hidden_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    // Run GEMV with first layer wq
    if let Some(layer) = ctx.model.layers.first() {
        if let Some(wq) = &layer.wq {
            let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
            let out_size = wq.shape[1];

            let output_buffer = Rc::new(gpu_ctx.device().new_buffer(
                (out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let cmd = gpu_ctx.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            gpu_ctx.mul_mv_q5_k_encode(
                &enc,
                &wq_buffer,
                &input_buffer,
                &output_buffer,
                out_size,
                hidden_size,
            );

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Check output
            let out_ptr = output_buffer.contents() as *const f32;
            let output: Vec<f32> = unsafe {
                std::slice::from_raw_parts(out_ptr, out_size).to_vec()
            };

            let has_nan = output.iter().any(|&x| x.is_nan());
            let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            println!("\nGEMV output (1 token):");
            println!("  out_size: {}", out_size);
            println!("  NaN: {}, Min: {:.6}, Max: {:.6}", has_nan, min, max);

            if has_nan {
                println!("  ❌ GEMV produces NaN");
            } else {
                println!("  ✓ GEMV OK");
            }
        }
    }
}
