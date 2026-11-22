// Debug scale assignment

use half::f16;
use infer_train::backend::arm::vec_dot::vec_dot_q5_k_q8_k;
use infer_train::types::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn create_deterministic_q5k_block(seed: u64) -> BlockQ5_K {
    let mut rng = StdRng::seed_from_u64(seed);
    let d = rng.gen_range(0.5..1.0);
    let dmin = rng.gen_range(0.05..0.1);
    let mut scales = [0u8; 12];
    let mut qh = [0u8; 32];
    let mut qs = [0u8; 128];

    for i in 0..12 {
        scales[i] = rng.gen_range(0..32);
    }
    for i in 0..32 {
        qh[i] = rng.gen_range(0..256) as u8;
    }
    for i in 0..128 {
        qs[i] = rng.gen_range(0..256) as u8;
    }

    BlockQ5_K {
        d: f16::from_f32(d).to_bits(),
        dmin: f16::from_f32(dmin).to_bits(),
        scales,
        qh,
        qs,
    }
}

fn create_deterministic_q8k_block(seed: u64) -> BlockQ8_K {
    let mut rng = StdRng::seed_from_u64(seed + 1000);
    let d = rng.gen_range(0.5..1.0);
    let mut qs = [0i8; 256];
    let mut bsums = [0i16; 16];

    for i in 0..256 {
        qs[i] = rng.gen_range(-64..64);
    }
    for i in 0..16 {
        let mut sum = 0i32;
        for j in 0..16 {
            sum += qs[i * 16 + j] as i32;
        }
        bsums[i] = sum.clamp(-32768, 32767) as i16;
    }

    BlockQ8_K { d, qs, bsums }
}

/// Decode scales and mins from packed 12-byte format
fn decode_scales_mins(packed: &[u8; 12]) -> ([i32; 8], [i32; 8]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp: [u32; 4] = [0; 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            packed.as_ptr(),
            utmp.as_mut_ptr() as *mut u8,
            12,
        );
    }

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let scales_bytes: [u8; 8] =
        unsafe { std::ptr::read(utmp.as_ptr() as *const [u8; 8]) };
    let mins_bytes: [u8; 8] =
        unsafe { std::ptr::read(utmp[2..].as_ptr() as *const [u8; 8]) };

    let mut scales = [0i32; 8];
    let mut mins = [0i32; 8];
    for i in 0..8 {
        scales[i] = scales_bytes[i] as i32;
        mins[i] = mins_bytes[i] as i32;
    }
    (scales, mins)
}

#[test]
fn test_scale_assignment() {
    let q5k = create_deterministic_q5k_block(42);
    let q8k = create_deterministic_q8k_block(42);
    let (scales, mins) = decode_scales_mins(&q5k.scales);

    println!("Scales: {:?}", scales);
    println!("Mins: {:?}", mins);

    // Unpack Q5_K with the NEW interleaved layout
    let mut q = [0u8; 256];
    for j in 0..4 {
        let base = j * 64;
        let qs_base = j * 32;
        let bit_lo = j * 2;
        let bit_hi = j * 2 + 1;

        // Low nibbles with bit `bit_lo`
        for k in 0..16 {
            let lo = q5k.qs[qs_base + k] & 0x0F;
            let hi_bit = (q5k.qh[k] >> bit_lo) & 1;
            q[base + k] = lo | (hi_bit << 4);
        }
        for k in 0..16 {
            let lo = q5k.qs[qs_base + 16 + k] & 0x0F;
            let hi_bit = (q5k.qh[16 + k] >> bit_lo) & 1;
            q[base + 16 + k] = lo | (hi_bit << 4);
        }

        // High nibbles with bit `bit_hi`
        for k in 0..16 {
            let hi = q5k.qs[qs_base + k] >> 4;
            let hi_bit = (q5k.qh[k] >> bit_hi) & 1;
            q[base + 32 + k] = hi | (hi_bit << 4);
        }
        for k in 0..16 {
            let hi = q5k.qs[qs_base + 16 + k] >> 4;
            let hi_bit = (q5k.qh[16 + k] >> bit_hi) & 1;
            q[base + 48 + k] = hi | (hi_bit << 4);
        }
    }

    // Compute with scale assignments
    // scale[0] for q[0..32], scale[1] for q[32..64], etc.
    let mut sumi = 0i64;
    for s in 0..8 {
        let scale = scales[s] as i64;
        let min = mins[s] as i64;

        let mut raw = 0i64;
        for e in 0..32 {
            raw += q[s * 32 + e] as i64 * q8k.qs[s * 32 + e] as i64;
        }

        let bsum = q8k.bsums[s * 2] as i64 + q8k.bsums[s * 2 + 1] as i64;
        let corr = raw - 16 * bsum;

        println!(
            "s={}: raw={}, bsum={}, corr={}, scale={}, contribution={}",
            s,
            raw,
            bsum,
            corr,
            scale,
            scale * corr
        );

        sumi += scale * corr;
    }
    println!("Total sumi (before min correction): {}", sumi);

    // Min correction
    let mut mins_sum = 0i64;
    for s in 0..8 {
        let bsum = q8k.bsums[s * 2] as i64 + q8k.bsums[s * 2 + 1] as i64;
        mins_sum += mins[s] as i64 * bsum;
    }
    println!("Mins sum: {}", mins_sum);

    let weight_d = f16::to_f32(f16::from_bits(q5k.d));
    let weight_dmin = f16::to_f32(f16::from_bits(q5k.dmin));
    let act_d = q8k.d;

    let result =
        weight_d * act_d * sumi as f32 - weight_dmin * act_d * mins_sum as f32;
    let neon_result = vec_dot_q5_k_q8_k(&[q5k], &[q8k]);

    println!("\nMy result:   {}", result);
    println!("NEON result: {}", neon_result);
    println!("Diff: {}", (result - neon_result).abs());
}
