#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <algorithm>
#include <cmath>

namespace infer_train {

// ============================================================
// MaxPool1d
// ============================================================
template<typename T>
Tensor<T> maxpool1d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    if (input.shape.size() != 3) return Tensor<T>();
    if (stride == -1) stride = kernel_size;

    using Conv = DTypeConverter<T>;
    size_t N = input.shape[0], C = input.shape[1], L = input.shape[2];
    int L_out = static_cast<int>((L + 2 * padding - kernel_size) / stride + 1);
    if (L_out <= 0) return Tensor<T>();

    Tensor<T> output({N, C, (size_t)L_out});

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int l = 0; l < L_out; ++l) {
                int l_start = l * stride - padding;
                float max_val = -std::numeric_limits<float>::infinity();
                for (int kl = 0; kl < kernel_size; ++kl) {
                    int in_l = l_start + kl;
                    if (in_l < 0 || in_l >= (int)L) continue;
                    float val = Conv::to_float(input.data[n * C * L + c * L + in_l]);
                    if (val > max_val) max_val = val;
                }
                output.data[n * C * L_out + c * L_out + l] = Conv::from_float(max_val);
            }
        }
    }
    return output;
}

// ============================================================
// MaxPool3d
// ============================================================
template<typename T>
Tensor<T> maxpool3d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    if (input.shape.size() != 5) return Tensor<T>();
    if (stride == -1) stride = kernel_size;

    using Conv = DTypeConverter<T>;
    size_t N = input.shape[0], C = input.shape[1];
    size_t D = input.shape[2], H = input.shape[3], W = input.shape[4];

    int D_out = static_cast<int>((D + 2 * padding - kernel_size) / stride + 1);
    int H_out = static_cast<int>((H + 2 * padding - kernel_size) / stride + 1);
    int W_out = static_cast<int>((W + 2 * padding - kernel_size) / stride + 1);
    if (D_out <= 0 || H_out <= 0 || W_out <= 0) return Tensor<T>();

    Tensor<T> output({N, C, (size_t)D_out, (size_t)H_out, (size_t)W_out});

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int d = 0; d < D_out; ++d) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        int d_start = d * stride - padding;
                        int h_start = h * stride - padding;
                        int w_start = w * stride - padding;
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int kd = 0; kd < kernel_size; ++kd) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int in_d = d_start + kd;
                                    int in_h = h_start + kh;
                                    int in_w = w_start + kw;
                                    if (in_d < 0 || in_d >= (int)D ||
                                        in_h < 0 || in_h >= (int)H ||
                                        in_w < 0 || in_w >= (int)W) continue;
                                    float val = Conv::to_float(
                                        input.data[n * C * D * H * W +
                                                   c * D * H * W +
                                                   in_d * H * W +
                                                   in_h * W +
                                                   in_w]
                                    );
                                    if (val > max_val) max_val = val;
                                }
                            }
                        }
                        output.data[n * C * D_out * H_out * W_out +
                                    c * D_out * H_out * W_out +
                                    d * H_out * W_out +
                                    h * W_out +
                                    w] = Conv::from_float(max_val);
                    }
                }
            }
        }
    }
    return output;
}

// ============================================================
// AvgPool1d
// ============================================================
template<typename T>
Tensor<T> avgpool1d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    if (input.shape.size() != 3) return Tensor<T>();
    if (stride == -1) stride = kernel_size;

    using Conv = DTypeConverter<T>;
    size_t N = input.shape[0], C = input.shape[1], L = input.shape[2];
    int L_out = static_cast<int>((L + 2 * padding - kernel_size) / stride + 1);
    if (L_out <= 0) return Tensor<T>();

    Tensor<T> output({N, C, (size_t)L_out});

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int l = 0; l < L_out; ++l) {
                int l_start = l * stride - padding;
                float sum = 0.0f; int count = 0;
                for (int kl = 0; kl < kernel_size; ++kl) {
                    int in_l = l_start + kl;
                    if (in_l < 0 || in_l >= (int)L) continue;
                    sum += Conv::to_float(input.data[n * C * L + c * L + in_l]);
                    count++;
                }
                output.data[n * C * L_out + c * L_out + l] = Conv::from_float(sum / count);
            }
        }
    }
    return output;
}

// ============================================================
// AvgPool3d
// ============================================================
template<typename T>
Tensor<T> avgpool3d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    if (input.shape.size() != 5) return Tensor<T>();
    if (stride == -1) stride = kernel_size;

    using Conv = DTypeConverter<T>;
    size_t N = input.shape[0], C = input.shape[1];
    size_t D = input.shape[2], H = input.shape[3], W = input.shape[4];

    int D_out = static_cast<int>((D + 2 * padding - kernel_size) / stride + 1);
    int H_out = static_cast<int>((H + 2 * padding - kernel_size) / stride + 1);
    int W_out = static_cast<int>((W + 2 * padding - kernel_size) / stride + 1);
    if (D_out <= 0 || H_out <= 0 || W_out <= 0) return Tensor<T>();

    Tensor<T> output({N, C, (size_t)D_out, (size_t)H_out, (size_t)W_out});

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int d = 0; d < D_out; ++d) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        int d_start = d * stride - padding;
                        int h_start = h * stride - padding;
                        int w_start = w * stride - padding;
                        float sum = 0.0f; int count = 0;
                        for (int kd = 0; kd < kernel_size; ++kd) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int in_d = d_start + kd;
                                    int in_h = h_start + kh;
                                    int in_w = w_start + kw;
                                    if (in_d < 0 || in_d >= (int)D ||
                                        in_h < 0 || in_h >= (int)H ||
                                        in_w < 0 || in_w >= (int)W) continue;
                                    sum += Conv::to_float(
                                        input.data[n * C * D * H * W +
                                                   c * D * H * W +
                                                   in_d * H * W +
                                                   in_h * W +
                                                   in_w]
                                    );
                                    count++;
                                }
                            }
                        }
                        output.data[n * C * D_out * H_out * W_out +
                                    c * D_out * H_out * W_out +
                                    d * H_out * W_out +
                                    h * W_out +
                                    w] = Conv::from_float(sum / count);
                    }
                }
            }
        }
    }
    return output;
}

} // namespace infer_train
