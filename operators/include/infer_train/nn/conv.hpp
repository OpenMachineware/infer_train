#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>
#include <cmath>

namespace infer_train {

// ============================================================
// Conv1d
// ============================================================
template<typename T>
Tensor<T> conv1d(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>* bias = nullptr,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1
) {
    if (input.shape.size() != 3 || weight.shape.size() != 3) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t L = input.shape[2];

    size_t O = weight.shape[0];
    size_t Cg = weight.shape[1];
    size_t KL = weight.shape[2];

    if (groups != 1 || C != Cg) return Tensor<T>();

    int L_out = static_cast<int>((L + 2 * padding - KL) / stride + 1);
    if (L_out <= 0) return Tensor<T>();

    Tensor<T> output({N, O, (size_t)L_out});

    for (size_t n = 0; n < N; ++n) {
        for (size_t o = 0; o < O; ++o) {
            for (int l = 0; l < L_out; ++l) {
                int l_start = l * stride - padding;
                float sum = 0.0f;
                for (size_t c = 0; c < C; ++c) {
                    for (size_t kl = 0; kl < KL; ++kl) {
                        int in_l = l_start + kl * dilation;
                        if (in_l < 0 || in_l >= (int)L) continue;
                        float inp = Conv::to_float(
                            input.data[n * C * L + c * L + in_l]
                        );
                        float wgt = Conv::to_float(
                            weight.data[o * C * KL + c * KL + kl]
                        );
                        sum += inp * wgt;
                    }
                }
                if (bias) sum += Conv::to_float(bias->data[o]);
                output.data[n * O * L_out + o * L_out + l] = Conv::from_float(sum);
            }
        }
    }
    return output;
}

// ============================================================
// Conv3d
// ============================================================
template<typename T>
Tensor<T> conv3d(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>* bias = nullptr,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1
) {
    if (input.shape.size() != 5 || weight.shape.size() != 5) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t D = input.shape[2];
    size_t H = input.shape[3];
    size_t W = input.shape[4];

    size_t O = weight.shape[0];
    size_t Cg = weight.shape[1];
    size_t KD = weight.shape[2];
    size_t KH = weight.shape[3];
    size_t KW = weight.shape[4];

    if (groups != 1 || C != Cg) return Tensor<T>();

    int D_out = static_cast<int>((D + 2 * padding - KD) / stride + 1);
    int H_out = static_cast<int>((H + 2 * padding - KH) / stride + 1);
    int W_out = static_cast<int>((W + 2 * padding - KW) / stride + 1);
    if (D_out <= 0 || H_out <= 0 || W_out <= 0) return Tensor<T>();

    Tensor<T> output({N, O, (size_t)D_out, (size_t)H_out, (size_t)W_out});

    for (size_t n = 0; n < N; ++n) {
        for (size_t o = 0; o < O; ++o) {
            for (int d = 0; d < D_out; ++d) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        int d_start = d * stride - padding;
                        int h_start = h * stride - padding;
                        int w_start = w * stride - padding;
                        float sum = 0.0f;
                        for (size_t c = 0; c < C; ++c) {
                            for (size_t kd = 0; kd < KD; ++kd) {
                                for (size_t kh = 0; kh < KH; ++kh) {
                                    for (size_t kw = 0; kw < KW; ++kw) {
                                        int in_d = d_start + kd * dilation;
                                        int in_h = h_start + kh * dilation;
                                        int in_w = w_start + kw * dilation;
                                        if (in_d < 0 || in_d >= (int)D ||
                                            in_h < 0 || in_h >= (int)H ||
                                            in_w < 0 || in_w >= (int)W) continue;
                                        float inp = Conv::to_float(
                                            input.data[n * C * D * H * W +
                                                       c * D * H * W +
                                                       in_d * H * W +
                                                       in_h * W +
                                                       in_w]
                                        );
                                        float wgt = Conv::to_float(
                                            weight.data[o * C * KD * KH * KW +
                                                        c * KD * KH * KW +
                                                        kd * KH * KW +
                                                        kh * KW +
                                                        kw]
                                        );
                                        sum += inp * wgt;
                                    }
                                }
                            }
                        }
                        if (bias) sum += Conv::to_float(bias->data[o]);
                        output.data[n * O * D_out * H_out * W_out +
                                    o * D_out * H_out * W_out +
                                    d * H_out * W_out +
                                    h * W_out +
                                    w] = Conv::from_float(sum);
                    }
                }
            }
        }
    }
    return output;
}

} // namespace infer_train
