#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <algorithm>
#include <cmath>

namespace infer_train {

// ============================================================
// MaxPool2d
// ============================================================
template<typename T>
Tensor<T> maxpool2d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    if (input.shape.size() != 4) {
        return Tensor<T>();
    }

    if (stride == -1) stride = kernel_size;

    using Conv = DTypeConverter<T>;

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t H = input.shape[2];
    size_t W = input.shape[3];

    int H_out = static_cast<int>((H + 2 * padding - kernel_size) / stride + 1);
    int W_out = static_cast<int>((W + 2 * padding - kernel_size) / stride + 1);

    if (H_out <= 0 || W_out <= 0) {
        return Tensor<T>();
    }

    Tensor<T> output({N, C, (size_t)H_out, (size_t)W_out});

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    int h_start = h * stride - padding;
                    int w_start = w * stride - padding;

                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int in_h = h_start + kh;
                            int in_w = w_start + kw;

                            if (in_h < 0 || in_h >= (int)H ||
                                in_w < 0 || in_w >= (int)W) {
                                continue;
                            }

                            float val = Conv::to_float(
                                input.data[n * C * H * W +
                                           c * H * W +
                                           in_h * W +
                                           in_w]
                            );
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }

                    output.data[n * C * H_out * W_out +
                                c * H_out * W_out +
                                h * W_out +
                                w] = Conv::from_float(max_val);
                }
            }
        }
    }

    return output;
}

// ============================================================
// AvgPool2d
// ============================================================
template<typename T>
Tensor<T> avgpool2d(
    const Tensor<T>& input,
    int kernel_size,
    int stride = -1,
    int padding = 0
) {
    if (input.shape.size() != 4) {
        return Tensor<T>();
    }

    if (stride == -1) stride = kernel_size;

    using Conv = DTypeConverter<T>;

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t H = input.shape[2];
    size_t W = input.shape[3];

    int H_out = static_cast<int>((H + 2 * padding - kernel_size) / stride + 1);
    int W_out = static_cast<int>((W + 2 * padding - kernel_size) / stride + 1);

    if (H_out <= 0 || W_out <= 0) {
        return Tensor<T>();
    }

    Tensor<T> output({N, C, (size_t)H_out, (size_t)W_out});

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    int h_start = h * stride - padding;
                    int w_start = w * stride - padding;

                    float sum = 0.0f;
                    int count = 0;

                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int in_h = h_start + kh;
                            int in_w = w_start + kw;

                            if (in_h < 0 || in_h >= (int)H ||
                                in_w < 0 || in_w >= (int)W) {
                                continue;
                            }

                            sum += Conv::to_float(
                                input.data[n * C * H * W +
                                           c * H * W +
                                           in_h * W +
                                           in_w]
                            );
                            count++;
                        }
                    }

                    float avg = (count > 0) ? (sum / count) : 0.0f;
                    output.data[n * C * H_out * W_out +
                                c * H_out * W_out +
                                h * W_out +
                                w] = Conv::from_float(avg);
                }
            }
        }
    }

    return output;
}

} // namespace infer_train
