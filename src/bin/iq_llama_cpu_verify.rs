// Direct comparison: Our NEON vec_dot vs llama.cpp's vec_dot
// Both use identical IQ4_NL weights and Q8_0 input

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::vec_dot_iq4_nl_q8_0_neon;
use rand::{rngs::StdRng, Rng, SeedableRng};

// Link to llama.cpp's native implementation
unsafe extern "C" {
    fn llama_vec_dot_iq4_nl_q8_0(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
}

fn main() {
    println!("=== NEON vs llama.cpp CPU Direct Comparison ===\n");

    let k = 4096usize;
    let nb = k / 32; // 128 blocks
    let seed = 42u64;

    let mut rng = StdRng::seed_from_u64(seed);

    // Generate IQ4_NL weights
    let weights: Vec<BlockIQ4NL> = (0..nb) // Single row
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect();

    // Generate Q8_0 input
    let input_f32: Vec<f32> =
        (0..k).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect();

    // Quantize to Q8_0
    let mut q8_blocks: Vec<BlockQ8_0> = Vec::with_capacity(nb);
    for b in 0..nb {
        let block_vals = &input_f32[b * 32..(b + 1) * 32];
        let amax = block_vals.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        let id = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        let mut qs = [0i8; 32];
        for (i, &v) in block_vals.iter().enumerate() {
            qs[i] = (v * id).round() as i8;
        }
        q8_blocks.push(BlockQ8_0 { d: f16::from_f32(scale).to_bits(), qs });
    }

    println!("Testing single row with K={}", k);
    println!("IQ4_NL blocks: {} ({} bytes)", nb, nb * 18);
    println!("Q8_0 blocks: {} ({} bytes)", nb, nb * 34);
    println!();

    // Our NEON implementation
    let our_result =
        unsafe { vec_dot_iq4_nl_q8_0_neon(k, &weights, &q8_blocks) };

    // llama.cpp CPU implementation
    let mut llama_result = 0.0f32;
    unsafe {
        llama_vec_dot_iq4_nl_q8_0(
            k as i32,
            &mut llama_result as *mut f32,
            weights.as_ptr() as *const u8,
            q8_blocks.as_ptr() as *const u8,
        );
    }

    let diff = (our_result - llama_result).abs();
    let rel_diff = if llama_result != 0.0 {
        diff / llama_result.abs() * 100.0
    } else {
        0.0
    };

    println!("{:<20} {:>15}", "Implementation", "Result");
    println!("{}", "-".repeat(40));
    println!("{:<20} {:>15.4}", "Our NEON", our_result);
    println!("{:<20} {:>15.4}", "llama.cpp CPU", llama_result);
    println!();
    println!("Difference: {:.4}", diff);
    println!("Relative:   {:.2}%", rel_diff);

    if diff < 0.01 || rel_diff < 1.0 {
        println!("\n✅ Results match! Our NEON = llama.cpp CPU");
    } else {
        println!("\n⚠️  Results differ. Investigating...");
    }

    // Also verify with direct calculation
    let kvalues: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0,
        25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
    ];

    let mut direct_sum = 0.0f32;
    for b in 0..nb {
        let scale = f16::from_bits(weights[b].d).to_f32();
        let q8_scale = f16::from_bits(q8_blocks[b].d).to_f32();
        for i in 0..16 {
            let qs = weights[b].qs[i];
            let lo_nibble = (qs & 0x0F) as usize;
            let hi_nibble = ((qs >> 4) & 0x0F) as usize;
            let q8_lo = q8_blocks[b].qs[2 * i] as f32;
            let q8_hi = q8_blocks[b].qs[2 * i + 1] as f32;
            direct_sum += scale
                * q8_scale
                * (q8_lo * kvalues[lo_nibble] + q8_hi * kvalues[hi_nibble]);
        }
    }

    println!("\nDirect calculation: {:.4}", direct_sum);
}
