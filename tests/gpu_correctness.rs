#[cfg(target_arch = "aarch64")]
#[test]
fn test_gpu_correctness() {
    use infer_train::quant::types::{BlockIQ4NL, BlockQ8_0};
    use infer_train::quant::vec_dot::arm::vec_dot_iq4_nl_q8_0_neon;
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use half::f16;

    let nb = 64;
    let n = nb * 32;

    // Generate test data
    let mut x_blocks = Vec::with_capacity(nb);
    let mut y_blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        x_blocks.push(BlockIQ4NL {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        });

        let mut y_qs = [0i8; 32];
        for j in 0..32 {
            let val = ((i * 23 + j * 7) % 256) as i32 - 128;
            y_qs[j] = val as i8;
        }
        y_blocks.push(BlockQ8_0 {
            d: f16::from_f32(1.0 + (i % 10) as f32 * 0.1).to_bits(),
            qs: y_qs,
        });
    }

    // CPU result
    let cpu_result = unsafe { vec_dot_iq4_nl_q8_0_neon(n, &x_blocks, &y_blocks) };

    // GPU result
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let gpu_result = ctx.vec_dot_iq4_nl_q8_0(n, &x_blocks, &y_blocks).expect("GPU failed");

    println!("CPU result: {}", cpu_result);
    println!("GPU result: {}", gpu_result);

    if cpu_result.is_nan() || gpu_result.is_nan() {
        panic!("NaN detected: cpu={}, gpu={}", cpu_result, gpu_result);
    }

    let diff = (cpu_result - gpu_result).abs();
    println!("Difference: {}", diff);

    // Allow small floating point difference
    let rel_error = diff / cpu_result.abs().max(1e-10);
    assert!(rel_error < 1e-4, "GPU result differs from CPU: {} vs {}, rel_error={}",
            cpu_result, gpu_result, rel_error);
}
