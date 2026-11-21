// Test single block vec_dot with detailed debug

use half::f16;
use infer_train::backend::arm::tiled::{
    TiledTileSrc0, TiledTileSrc1, decode_q5k_scales, tiled_unpack_src0_q5_k,
    tiled_unpack_src1_q8_k, tiled_unpk_nib4, tiled_unpk_or,
};
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

#[test]
fn test_q5k_unpack_and_compute() {
    let q5k_block = create_deterministic_q5k_block(42);
    let q8k_block = create_deterministic_q8k_block(42);

    println!("=== BlockQ5_K ===");
    println!(
        "d: {} (f16 bits: {})",
        f16::to_f32(f16::from_bits(q5k_block.d)),
        q5k_block.d
    );
    println!(
        "dmin: {} (f16 bits: {})",
        f16::to_f32(f16::from_bits(q5k_block.dmin)),
        q5k_block.dmin
    );
    println!("scales: {:?}", &q5k_block.scales[..]);
    println!("qh[0..8]: {:?}", &q5k_block.qh[0..8]);
    println!("qs[0..8]: {:?}", &q5k_block.qs[0..8]);

    // Decode scales/mins - debug step by step
    let packed = &q5k_block.scales;
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
    println!("utmp initial: {:?}", utmp);

    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    println!("utmp after decode: {:?}", utmp);

    let scales_bytes: [u8; 8] =
        unsafe { std::ptr::read(utmp.as_ptr() as *const [u8; 8]) };
    let mins_bytes: [u8; 8] =
        unsafe { std::ptr::read(utmp[2..].as_ptr() as *const [u8; 8]) };

    println!("scales_bytes: {:?}", scales_bytes);
    println!("mins_bytes: {:?}", mins_bytes);

    let (scales, mins) = decode_q5k_scales(&q5k_block.scales);
    println!("Decoded scales: {:?}", scales);
    println!("Decoded mins: {:?}", mins);

    println!("\n=== BlockQ8_K ===");
    println!("d: {}", q8k_block.d);
    println!("qs[0..16]: {:?}", &q8k_block.qs[0..16]);
    println!("bsums: {:?}", q8k_block.bsums);

    // Compute with original vec_dot
    let orig_result = vec_dot_q5_k_q8_k(&[q5k_block], &[q8k_block]);
    println!("\n=== Original vec_dot result: {} ===", orig_result);

    // Manual unpack test
    println!("\n=== Manual unpack test ===");
    let mut q_unpacked = [0u8; 256];

    // Unpack nibbles - use temporary buffers
    {
        let (lo, hi) = q_unpacked[0..64].split_at_mut(32);
        tiled_unpk_nib4(&q5k_block.qs[0..32], lo, hi);
    }
    {
        let (lo, hi) = q_unpacked[64..128].split_at_mut(32);
        tiled_unpk_nib4(&q5k_block.qs[32..64], lo, hi);
    }
    {
        let (lo, hi) = q_unpacked[128..192].split_at_mut(32);
        tiled_unpk_nib4(&q5k_block.qs[64..96], lo, hi);
    }
    {
        let (lo, hi) = q_unpacked[192..256].split_at_mut(32);
        tiled_unpk_nib4(&q5k_block.qs[96..128], lo, hi);
    }

    // Add high bit
    tiled_unpk_or::<0, 4, 1>(&mut q_unpacked[0..32], &q5k_block.qh);
    tiled_unpk_or::<1, 4, 1>(&mut q_unpacked[32..64], &q5k_block.qh);
    tiled_unpk_or::<2, 4, 1>(&mut q_unpacked[64..96], &q5k_block.qh);
    tiled_unpk_or::<3, 4, 1>(&mut q_unpacked[96..128], &q5k_block.qh);
    tiled_unpk_or::<4, 4, 1>(&mut q_unpacked[128..160], &q5k_block.qh);
    tiled_unpk_or::<5, 4, 1>(&mut q_unpacked[160..192], &q5k_block.qh);
    tiled_unpk_or::<6, 4, 1>(&mut q_unpacked[192..224], &q5k_block.qh);
    tiled_unpk_or::<7, 4, 1>(&mut q_unpacked[224..256], &q5k_block.qh);

    println!("Unpacked q[0..16]: {:?}", &q_unpacked[0..16]);
    println!("Unpacked q[32..48]: {:?}", &q_unpacked[32..48]);

    // Manual dot product calculation (subblock 0)
    let weight_d = f16::to_f32(f16::from_bits(q5k_block.d));
    let weight_dmin = f16::to_f32(f16::from_bits(q5k_block.dmin));
    let act_d = q8k_block.d;

    let mut total_sumi = 0i64;
    let mut total_mins_bsum = 0i64;

    for s in 0..8 {
        let scale = scales[s] as i64;
        let min = mins[s] as i64;

        // Compute raw dot for this subblock
        let mut raw = 0i64;
        for k in 0..32 {
            let w = q_unpacked[s * 32 + k] as i64;
            let a = q8k_block.qs[s * 32 + k] as i64;
            raw += w * a;
        }

        // Bias correction: subtract 16 * bsum
        let bsum0 = q8k_block.bsums[s * 2] as i64;
        let bsum1 = q8k_block.bsums[s * 2 + 1] as i64;
        let bsum = bsum0 + bsum1;

        let corr = raw - 16 * bsum;
        let scaled = scale * corr;

        // Min correction
        let mins_bsum = min * bsum;

        println!(
            "Subblock {}: scale={}, min={}, raw={}, bsum={}, corr={}, scaled={}, mins_bsum={}",
            s, scale, min, raw, bsum, corr, scaled, mins_bsum
        );

        total_sumi += scaled;
        total_mins_bsum += mins_bsum;
    }

    let d = weight_d * act_d;
    let dmin = weight_dmin * act_d;

    let result = d * total_sumi as f32 - dmin * total_mins_bsum as f32;
    println!(
        "\nManual calculation: d={}, dmin={}, sumi={}, mins_bsum={}",
        d, dmin, total_sumi, total_mins_bsum
    );
    println!("Manual result: {}", result);
    println!("Original result: {}", orig_result);
    println!("Difference: {}", (result - orig_result).abs());
}
