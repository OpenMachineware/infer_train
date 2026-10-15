// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test IQ4_XS GPU kernel correctness

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod test_metal {
    use infer_train::quant::types::BlockIQ4XS;
    use metal::*;
    use std::ffi::c_void;

    const QK_K: usize = 256;

    #[repr(C)]
    struct MulMVArgs {
        ne00: i32,
        ne01: i32,
        ne02: i32,
        nb00: u64,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne10: i32,
        ne11: i32,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        nr0: i32,
        r2: i16,
        r3: i16,
    }

    fn create_function_constants(nsg: i16) -> FunctionConstantValues {
        const FC_MUL_MV: NSUInteger = 600;
        let cv = FunctionConstantValues::new();
        unsafe {
            cv.set_constant_value_at_index(
                &nsg as *const i16 as *const c_void,
                MTLDataType::Short,
                FC_MUL_MV + 0,
            );
            let ne12: i16 = 1;
            let r2: i16 = 1;
            let r3: i16 = 1;
            cv.set_constant_value_at_index(
                &ne12 as *const i16 as *const c_void,
                MTLDataType::Short,
                FC_MUL_MV + 2,
            );
            cv.set_constant_value_at_index(
                &r2 as *const i16 as *const c_void,
                MTLDataType::Short,
                FC_MUL_MV + 3,
            );
            cv.set_constant_value_at_index(
                &r3 as *const i16 as *const c_void,
                MTLDataType::Short,
                FC_MUL_MV + 4,
            );
        }
        cv
    }

    fn dequant_iq4_xs_cpu(data: &[u8], out: &mut [f32], count: usize) {
        // IQ4_NL grid values (kvalues_iq4nl)
        const KVALUES: [f32; 16] = [
            -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0,
            13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
        ];

        let n_blocks = count.div_ceil(QK_K);
        for block_idx in 0..n_blocks {
            let block_offset = block_idx * std::mem::size_of::<BlockIQ4XS>();
            if block_offset + std::mem::size_of::<BlockIQ4XS>() > data.len() {
                break;
            }
            let block: &BlockIQ4XS = unsafe {
                &*(data.as_ptr().add(block_offset) as *const BlockIQ4XS)
            };

            let d = half::f16::from_bits(block.d).to_f32();

            // Process 8 sub-blocks of 32 elements each
            for ib in 0..8 {
                // Extract scale for this sub-block (6-bit value from scales_l + scales_h)
                let ls = ((block.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF)
                    as i32
                    | ((((block.scales_h >> (2 * ib)) & 3) << 4) as i32);
                let dl = d * (ls - 32) as f32;

                // Each sub-block has 32 elements stored in 16 bytes
                for j in 0..16 {
                    let byte = block.qs[ib * 16 + j];
                    let idx0 = (byte & 0x0F) as usize;
                    let idx1 = ((byte >> 4) & 0x0F) as usize;

                    let out_idx0 = block_idx * QK_K + ib * 32 + j;
                    let out_idx1 = block_idx * QK_K + ib * 32 + j + 16;

                    if out_idx0 < count {
                        out[out_idx0] = KVALUES[idx0] * dl;
                    }
                    if out_idx1 < count {
                        out[out_idx1] = KVALUES[idx1] * dl;
                    }
                }
            }
        }
    }

    #[test]
    fn test_iq4_xs_gpu_correctness() {
        let device = Device::system_default().unwrap();
        let queue = device.new_command_queue();

        // Load our shader - only the kernel we need
        // We need to provide a minimal shader with just the IQ4_XS kernel
        let shader = r#"
using namespace metal;

#define QK_K 256

// IQ4_NL kvalues
constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

struct block_iq4_xs {
    half d;
    ushort scales_h;
    uchar scales_l[4];
    uchar qs[128];
};

struct mv_args {
    int ne00; int ne01; int ne02;
    ulong nb00; ulong nb01; ulong nb02; ulong nb03;
    int ne10; int ne11; int ne12;
    ulong nb10; ulong nb11; ulong nb12; ulong nb13;
    int ne0; int ne1; int nr0;
    short r2; short r3;
};

#define FOR_UNROLL for

kernel void kernel_mul_mv_iq4_xs_f32(
    device const char * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device float * dst [[buffer(2)]],
    constant mv_args & args [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short NSG = 2;
    const short NR0 = 2;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    if (first_row >= args.ne01) return;

    device const block_iq4_xs * x = (device const block_iq4_xs *)(src0 + first_row * args.nb01);

    const short ix = tiisg/16;
    const short it = tiisg%16;
    const short ib = it/2;
    const short il = it%2;

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = src1 + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]};
            qf2 = {kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]};
            qf2 = {kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * QK_K;
    }

    device float * dst_f32 = dst + first_row;

    for (int row = 0; row < NR0 && first_row + row < args.ne01; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[row] = sum_all;
        }
    }
}
"#;

        let compile_options = CompileOptions::new();
        let lib = device
            .new_library_with_source(&shader, &compile_options)
            .expect("Failed to compile shader");

        let cv = create_function_constants(2); // NSG = 2 for IQ4_XS
        let kernel_name = "kernel_mul_mv_iq4_xs_f32";
        let func = lib
            .get_function(kernel_name, Some(cv))
            .expect("Failed to get kernel function");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .expect("Failed to create pipeline");

        // Create test data
        let m = 64; // output rows
        let k = 4096; // input dimension (16 blocks)
        let nb = k / QK_K;

        // Create random block data
        let mut blocks: Vec<BlockIQ4XS> = Vec::new();
        for _ in 0..m * nb {
            let mut block = BlockIQ4XS {
                d: half::f16::to_bits(half::f16::from_f32(0.5)),
                scales_h: 0x1555, // mid-range scales
                scales_l: [0x55; 4],
                qs: [0u8; 128],
            };
            // Fill with varied data
            for i in 0..128 {
                block.qs[i] = ((i * 17 + 13) % 256) as u8;
            }
            blocks.push(block);
        }

        // Create input vector
        let input: Vec<f32> =
            (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

        // Compute expected output on CPU
        let mut expected = vec![0.0f32; m];
        for row in 0..m {
            // Dequantize weights for this row
            let block_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    blocks[row * nb..(row + 1) * nb].as_ptr() as *const u8,
                    nb * std::mem::size_of::<BlockIQ4XS>(),
                )
            };
            let mut weights = vec![0.0f32; k];
            dequant_iq4_xs_cpu(block_bytes, &mut weights, k);

            // Compute dot product
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += weights[i] * input[i];
            }
            expected[row] = sum;
        }

        // Run GPU kernel
        let weights_size = m * std::mem::size_of::<BlockIQ4XS>();
        let weights_buf = device.new_buffer_with_data(
            blocks.as_ptr() as *const _,
            weights_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buf = device.new_buffer_with_data(
            input.as_ptr() as *const _,
            (k * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device
            .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

        let args = MulMVArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01: std::mem::size_of::<BlockIQ4XS>() as u64,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: (k * 4) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let NR0 = 2; // rows per SIMD group
        let NSG = 2; // SIMD groups per threadgroup
        let rows_per_tg = NR0 * NSG;
        let grid =
            MTLSize::new(((m + rows_per_tg - 1) / rows_per_tg) as u64, 1, 1);
        let tg = MTLSize::new((NSG * 32) as u64, 1, 1);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&weights_buf), 0); // src0 = weights
        enc.set_buffer(1, Some(&input_buf), 0); // src1 = input
        enc.set_buffer(2, Some(&output_buf), 0); // dst = output
        enc.set_bytes(
            3,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Read results
        let result: &[f32] = unsafe {
            std::slice::from_raw_parts(output_buf.contents() as *const f32, m)
        };

        // Compare
        println!("\n=== IQ4_XS GPU Correctness Test ===");
        println!("M={}, K={}", m, k);
        let mut max_diff = 0.0f32;
        let mut nonzero_count = 0;
        for i in 0..m {
            let diff = (result[i] - expected[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if result[i].abs() > 1e-6 {
                nonzero_count += 1;
            }
            if i < 5 {
                println!(
                    "row {}: GPU={:.6}, CPU={:.6}, diff={:.6}",
                    i, result[i], expected[i], diff
                );
            }
        }
        println!("Max diff: {:.6}", max_diff);
        println!("Non-zero results: {}/{}", nonzero_count, m);

        // Check that results are not all zeros
        assert!(nonzero_count > 0, "GPU kernel returned all zeros!");

        // Check correctness within tolerance
        for i in 0..m {
            let rel_err = if expected[i].abs() > 1e-6 {
                (result[i] - expected[i]).abs() / expected[i].abs()
            } else {
                result[i].abs()
            };
            assert!(
                rel_err < 0.01,
                "Mismatch at row {}: GPU={:.6}, CPU={:.6}, rel_err={:.6}",
                i,
                result[i],
                expected[i],
                rel_err
            );
        }

        println!("✅ IQ4_XS GPU kernel correctness verified!");
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
mod test_metal {
    #[test]
    fn test_iq4_xs_gpu_correctness() {
        println!("Test skipped: Metal not available on this platform");
    }
}
