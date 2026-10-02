// Experimental Q3_K kernel with branch-free optimization
// Key change: Replace conditional with bitwise arithmetic

[[host_name("kernel_mul_mv_q3_K_f32_opt")]]
kernel void kernel_mul_mv_q3_K_f32_opt(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    const short NSG = FC_nsg;
    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q3_K * x = (device const block_q3_K *)(src0 + offset0);
    device const float      * yy = (device const float      *)(src1 + offset1);

    float yl[32];

    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short ip = tid / 4;
    const short il = 2 * ((tid % 4) / 2);
    const short ir = tid % 2;
    const short l0 = 8 * ir;

    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};

    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2 * ip + il / 2];

    const short shift = 2 * il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4 * ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32 * ip + l0;
    const short y_offset = 128 * ip + 32 * il + l0;

    device const float * y1 = yy + ix * QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);

                // Branch-free optimization: use bitwise complement
                // (h & hm) ? 0 : y  =>  y * (1 - (h & hm != 0))
                // But Metal doesn't have bool->int, so use: y * (1 - min(1, (h & hm)))
                const ushort h_val = h[l/2];
                const float m0 = 1.0f - min(1.0f, (float)((h_val & hm[0]) != 0)));
                const float m1 = 1.0f - min(1.0f, (float)((h_val & hm[1]) != 0)));
                s3 += yl[l+0] * m0 + yl[l+1] * m1;

                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);

                const float m2 = 1.0f - min(1.0f, (float)((h_val & hm[2]) != 0)));
                const float m3 = 1.0f - min(1.0f, (float)((h_val & hm[3]) != 0)));
                s6 += yl[l+16] * m2 + yl[l+17] * m3;
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);

                const ushort h_val = h[l/2+8];
                const float m0 = 1.0f - min(1.0f, (float)((h_val & hm[0]) != 0)));
                const float m1 = 1.0f - min(1.0f, (float)((h_val & hm[1]) != 0)));
                s3 += yl[l+8] * m0 + yl[l+9] * m1;

                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);

                const float m2 = 1.0f - min(1.0f, (float)((h_val & hm[2]) != 0)));
                const float m3 = 1.0f - min(1.0f, (float)((h_val & hm[3]) != 0)));
                s6 += yl[l+24] * m2 + yl[l+25] * m3;
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01 / 2;
            h  += args.nb01 / 2;
            a  += args.nb01 / 2;
            dh += args.nb01 / 2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < nr0; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    if (tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            dst_f32[first_row + row] = sumf1[row];
        }
    }
}
