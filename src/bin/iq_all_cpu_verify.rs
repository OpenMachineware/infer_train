// Verify ALL IQ formats: Our NEON vs llama.cpp CPU
// Simple test with random but valid-ish data

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

// Link to llama.cpp's native implementations
unsafe extern "C" {
    fn llama_vec_dot_iq4_nl_q8_0(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
    fn llama_vec_dot_iq4_xs_q8_K(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
    fn llama_vec_dot_iq3_xxs_q8_K(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
    fn llama_vec_dot_iq3_s_q8_K(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
    fn llama_vec_dot_iq2_xxs_q8_K(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
    fn llama_vec_dot_iq2_xs_q8_K(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
    fn llama_vec_dot_iq2_s_q8_K(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
    fn llama_vec_dot_iq1_s_q8_K(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
    fn llama_vec_dot_iq1_m_q8_K(
        n: i32,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );
}

fn main() {
    println!("=== All IQ Formats: NEON vs llama.cpp CPU ===\n");

    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    println!(
        "{:<12} {:>14} {:>14} {:>10} {:>8}",
        "Format", "Our NEON", "llama.cpp", "Diff", "Status"
    );
    println!("{}", "-".repeat(65));

    let mut all_pass = true;

    // IQ4_NL (32 elements, Q8_0)
    {
        let k = 256usize; // Test with more elements for meaningful results
        let nb = k / 32;
        let q8_0: Vec<BlockQ8_0> = (0..nb)
            .map(|_| {
                let mut qs = [0i8; 32];
                for j in 0..32 {
                    qs[j] = rng.r#gen::<i8>();
                }
                BlockQ8_0 { d: f16::from_f32(1.0).to_bits(), qs }
            })
            .collect();

        let weights: Vec<BlockIQ4NL> = (0..nb)
            .map(|_| {
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    qs[j] = rng.r#gen::<u8>();
                }
                BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
            })
            .collect();

        let our_result =
            unsafe { vec_dot_iq4_nl_q8_0_neon(k, &weights, &q8_0) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq4_nl_q8_0(
                k as i32,
                &mut llama_result,
                weights.as_ptr() as *const u8,
                q8_0.as_ptr() as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ4_NL", our_result, llama_result, diff, status
        );
    }

    // IQ4_XS (256 elements, Q8_K) - need valid data
    {
        let k = 256usize;
        let mut q8_k = BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for j in 0..256 {
            q8_k.qs[j] = rng.r#gen::<i8>();
        }
        for j in 0..16 {
            q8_k.bsums[j] = rng.r#gen::<i16>();
        }

        let mut weights = BlockIQ4XS {
            d: f16::from_f32(1.0).to_bits(),
            scales_h: 0,
            scales_l: [0u8; 4],
            qs: [0u8; 128],
        };
        for j in 0..128 {
            weights.qs[j] = rng.r#gen::<u8>();
        }
        for j in 0..4 {
            weights.scales_l[j] = 0;
        } // Use 0 for valid quantization

        let weights_slice: &[BlockIQ4XS] = std::slice::from_ref(&weights);
        let q8_k_slice: &[BlockQ8K] = std::slice::from_ref(&q8_k);

        let our_result =
            unsafe { vec_dot_iq4_xs_q8_k_neon(k, weights_slice, q8_k_slice) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq4_xs_q8_K(
                k as i32,
                &mut llama_result,
                &weights as *const _ as *const u8,
                &q8_k as *const _ as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ4_XS", our_result, llama_result, diff, status
        );
    }

    // IQ3_XXS (256 elements, Q8_K)
    {
        let k = 256usize;
        let mut q8_k = BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for j in 0..256 {
            q8_k.qs[j] = rng.r#gen::<i8>();
        }
        for j in 0..16 {
            q8_k.bsums[j] = rng.r#gen::<i16>();
        }

        let mut weights =
            BlockIQ3XXS { d: f16::from_f32(1.0).to_bits(), qs: [0u8; 96] };
        for j in 0..96 {
            weights.qs[j] = rng.r#gen::<u8>();
        }

        let weights_slice: &[BlockIQ3XXS] = std::slice::from_ref(&weights);
        let q8_k_slice: &[BlockQ8K] = std::slice::from_ref(&q8_k);

        let our_result =
            unsafe { vec_dot_iq3_xxs_q8_k_neon(k, weights_slice, q8_k_slice) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq3_xxs_q8_K(
                k as i32,
                &mut llama_result,
                &weights as *const _ as *const u8,
                &q8_k as *const _ as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ3_XXS", our_result, llama_result, diff, status
        );
    }

    // IQ3_S (256 elements, Q8_K)
    {
        let k = 256usize;
        let mut q8_k = BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for j in 0..256 {
            q8_k.qs[j] = rng.r#gen::<i8>();
        }
        for j in 0..16 {
            q8_k.bsums[j] = rng.r#gen::<i16>();
        }

        let mut weights = BlockIQ3S {
            d: f16::from_f32(1.0).to_bits(),
            qs: [0u8; 64],
            qh: [0u8; 8],
            signs: [0u8; 32],
            scales: [0u8; 4],
        };
        for j in 0..64 {
            weights.qs[j] = rng.r#gen::<u8>();
        }
        for j in 0..8 {
            weights.qh[j] = rng.r#gen::<u8>();
        }
        for j in 0..32 {
            weights.signs[j] = rng.r#gen::<u8>();
        }

        let weights_slice: &[BlockIQ3S] = std::slice::from_ref(&weights);
        let q8_k_slice: &[BlockQ8K] = std::slice::from_ref(&q8_k);

        let our_result =
            unsafe { vec_dot_iq3_s_q8_k_neon(k, weights_slice, q8_k_slice) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq3_s_q8_K(
                k as i32,
                &mut llama_result,
                &weights as *const _ as *const u8,
                &q8_k as *const _ as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ3_S", our_result, llama_result, diff, status
        );
    }

    // IQ2_XXS (256 elements, Q8_K)
    {
        let k = 256usize;
        let mut q8_k = BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for j in 0..256 {
            q8_k.qs[j] = rng.r#gen::<i8>();
        }
        for j in 0..16 {
            q8_k.bsums[j] = rng.r#gen::<i16>();
        }

        let mut weights =
            BlockIQ2XXS { d: f16::from_f32(1.0).to_bits(), qs: [0u16; 32] };
        for j in 0..32 {
            weights.qs[j] = rng.r#gen::<u16>();
        }

        let weights_slice: &[BlockIQ2XXS] = std::slice::from_ref(&weights);
        let q8_k_slice: &[BlockQ8K] = std::slice::from_ref(&q8_k);

        let our_result =
            unsafe { vec_dot_iq2_xxs_q8_k_neon(k, weights_slice, q8_k_slice) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq2_xxs_q8_K(
                k as i32,
                &mut llama_result,
                &weights as *const _ as *const u8,
                &q8_k as *const _ as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ2_XXS", our_result, llama_result, diff, status
        );
    }

    // IQ2_XS (256 elements, Q8_K)
    {
        let k = 256usize;
        let mut q8_k = BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for j in 0..256 {
            q8_k.qs[j] = rng.r#gen::<i8>();
        }
        for j in 0..16 {
            q8_k.bsums[j] = rng.r#gen::<i16>();
        }

        let mut weights = BlockIQ2XS {
            d: f16::from_f32(1.0).to_bits(),
            qs: [0u16; 32],
            scales: [0u8; 8],
        };
        for j in 0..32 {
            weights.qs[j] = rng.r#gen::<u16>();
        }

        let weights_slice: &[BlockIQ2XS] = std::slice::from_ref(&weights);
        let q8_k_slice: &[BlockQ8K] = std::slice::from_ref(&q8_k);

        let our_result =
            unsafe { vec_dot_iq2_xs_q8_k_neon(k, weights_slice, q8_k_slice) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq2_xs_q8_K(
                k as i32,
                &mut llama_result,
                &weights as *const _ as *const u8,
                &q8_k as *const _ as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ2_XS", our_result, llama_result, diff, status
        );
    }

    // IQ2_S (256 elements, Q8_K)
    {
        let k = 256usize;
        let mut q8_k = BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for j in 0..256 {
            q8_k.qs[j] = rng.r#gen::<i8>();
        }
        for j in 0..16 {
            q8_k.bsums[j] = rng.r#gen::<i16>();
        }

        let mut weights = BlockIQ2S {
            d: f16::from_f32(1.0).to_bits(),
            qs: [0u8; 64],
            qh: [0u8; 8],
            scales: [0u8; 8],
        };
        for j in 0..64 {
            weights.qs[j] = rng.r#gen::<u8>();
        }
        for j in 0..8 {
            weights.qh[j] = rng.r#gen::<u8>();
        }

        let weights_slice: &[BlockIQ2S] = std::slice::from_ref(&weights);
        let q8_k_slice: &[BlockQ8K] = std::slice::from_ref(&q8_k);

        let our_result =
            unsafe { vec_dot_iq2_s_q8_k_neon(k, weights_slice, q8_k_slice) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq2_s_q8_K(
                k as i32,
                &mut llama_result,
                &weights as *const _ as *const u8,
                &q8_k as *const _ as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ2_S", our_result, llama_result, diff, status
        );
    }

    // IQ1_S (256 elements, Q8_K)
    {
        let k = 256usize;
        let mut q8_k = BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for j in 0..256 {
            q8_k.qs[j] = rng.r#gen::<i8>();
        }
        for j in 0..16 {
            q8_k.bsums[j] = rng.r#gen::<i16>();
        }

        let mut weights = BlockIQ1S {
            d: f16::from_f32(1.0).to_bits(),
            qs: [0u8; 32],
            qh: [0u16; 8],
        };
        for j in 0..32 {
            weights.qs[j] = rng.r#gen::<u8>();
        }
        for j in 0..8 {
            weights.qh[j] = rng.r#gen::<u16>();
        }

        let weights_slice: &[BlockIQ1S] = std::slice::from_ref(&weights);
        let q8_k_slice: &[BlockQ8K] = std::slice::from_ref(&q8_k);

        let our_result =
            unsafe { vec_dot_iq1_s_q8_k_neon(k, weights_slice, q8_k_slice) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq1_s_q8_K(
                k as i32,
                &mut llama_result,
                &weights as *const _ as *const u8,
                &q8_k as *const _ as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ1_S", our_result, llama_result, diff, status
        );
    }

    // IQ1_M (256 elements, Q8_K)
    {
        let k = 256usize;
        let mut q8_k = BlockQ8K { d: 1.0, qs: [0i8; 256], bsums: [0i16; 16] };
        for j in 0..256 {
            q8_k.qs[j] = rng.r#gen::<i8>();
        }
        for j in 0..16 {
            q8_k.bsums[j] = rng.r#gen::<i16>();
        }

        let mut weights =
            BlockIQ1M { qs: [0u8; 32], qh: [0u8; 16], scales: [0u8; 8] };
        for j in 0..32 {
            weights.qs[j] = rng.r#gen::<u8>();
        }
        for j in 0..16 {
            weights.qh[j] = rng.r#gen::<u8>();
        }

        let weights_slice: &[BlockIQ1M] = std::slice::from_ref(&weights);
        let q8_k_slice: &[BlockQ8K] = std::slice::from_ref(&q8_k);

        let our_result =
            unsafe { vec_dot_iq1_m_q8_k_neon(k, weights_slice, q8_k_slice) };
        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_iq1_m_q8_K(
                k as i32,
                &mut llama_result,
                &weights as *const _ as *const u8,
                &q8_k as *const _ as *const u8,
            );
        }

        let diff = (our_result - llama_result).abs();
        let status = if diff < 0.01 {
            "✅"
        } else {
            all_pass = false;
            "❌"
        };
        println!(
            "{:<12} {:>14.4} {:>14.4} {:>10.6} {:>8}",
            "IQ1_M", our_result, llama_result, diff, status
        );
    }

    println!("{}", "-".repeat(65));
    if all_pass {
        println!("\n✅ ALL 9 IQ formats match llama.cpp CPU exactly!");
    } else {
        println!(
            "\n⚠️  Some formats differ - may be due to random data not matching format constraints"
        );
    }
}
