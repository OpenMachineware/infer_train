// Optimized dequantization functions for Q2_K and Q3_K
// Uses lookup tables instead of branches to avoid SIMD divergence

// Q2_K optimization: precomputed tables
constant const half q2_k_coef_table[4] = {1.0h, 0.25h, 0.0625h, 0.015625h}; // 1, 1/4, 1/16, 1/64
constant const uchar q2_k_mask_table[4] = {3, 12, 48, 192};

template <typename type4x4>
void dequantize_q2_K_opt(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;

    // Use lookup table instead of branches
    const uint8_t sc = xb->scales[il];
    const short il_shifted = (il/2) % 4;

    q = q + 32*(il/8) + 16*(il&1);

    // Table lookup - no branches
    const half coef = q2_k_coef_table[il_shifted];
    const uchar mask = q2_k_mask_table[il_shifted];

    const float dl = d * (sc & 0xF) * coef;
    const float ml = min * (sc >> 4);

    // Unrolled loop for better performance
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// Q3_K optimization: use lookup tables
constant const half q3_k_coef_table[4] = {1.0h, 0.25h, 0.0625h, 0.015625h};
constant const uchar q3_k_mask_table[4] = {3, 12, 48, 192};

template <typename type4x4>
void dequantize_q3_K_opt(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);

    const uint8_t m = 1 << (il/2);

    // Precompute scale indices and masks
    const short scale_idx_2 = il % 8;
    const short scale_idx_1 = 8 + (il % 4);

    // Compute scale using table lookup
    const uint16_t kmask1_table[4] = {3, 12, 48, 192};
    const uint16_t kmask2 = (il/8) ? 0xF0 : 0x0F;
    const uint16_t kmask1 = kmask1_table[(il/4) % 4];

    const uint16_t scale_2 = scales[scale_idx_2];
    const uint16_t scale_1 = scales[scale_idx_1];

    const int16_t dl_int = ((il/4) & 1)
        ? ((scale_2 & kmask2) | ((scale_1 & kmask1) << 2))
        : ((scale_2 & kmask2) | ((scale_1 & kmask1) << 4));

    const float dl_raw = (il < 8) ? (dl_int - 32.f) : (dl_int / 16.f - 32.f);
    float dl = d_all * dl_raw;
    const float ml = 4.f * dl;

    const short il_shifted = (il/2) & 3;
    dl *= q3_k_coef_table[il_shifted];
    const uchar mask = q3_k_mask_table[il_shifted];

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}
