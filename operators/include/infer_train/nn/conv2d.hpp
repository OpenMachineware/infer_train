#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>
#include <cmath>

namespace infer_train {

// ============================================================
// Conv2d 滑动窗口实现
// ============================================================
template<typename T>
Tensor<T> conv2d(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>* bias = nullptr,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1
) {
    // 输入: (N, C, H, W)
    // weight: (O, C/g, KH, KW)
    // bias: (O,) 可选

    if (input.shape.size() != 4 || weight.shape.size() != 4) {
        return Tensor<T>();  // 错误
    }

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t H = input.shape[2];
    size_t W = input.shape[3];

    size_t O = weight.shape[0];        // 输出通道数
    size_t Cg = weight.shape[1];       // 每个组的输入通道数
    size_t KH = weight.shape[2];
    size_t KW = weight.shape[3];

    // 检查 groups 合法性
    if (groups != 1) {
        // 分组卷积暂不支持，先返回空
        return Tensor<T>();
    }
    if (C != Cg) {
        return Tensor<T>();
    }

    // 计算输出尺寸
    int H_out = static_cast<int>((H + 2 * padding - KH) / stride + 1);
    int W_out = static_cast<int>((W + 2 * padding - KW) / stride + 1);

    if (H_out <= 0 || W_out <= 0) {
        return Tensor<T>();
    }

    // 使用 DTypeConverter
    using Conv = DTypeConverter<T>;
    Tensor<T> output({N, O, (size_t)H_out, (size_t)W_out});

    // 为了简化，先用 float 计算
    // 实际使用时需要 T 类型

    // 滑动窗口
    for (size_t n = 0; n < N; ++n) {
        for (size_t o = 0; o < O; ++o) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    // 计算当前窗口在输入上的起始位置
                    int h_start = h * stride - padding;
                    int w_start = w * stride - padding;

                    // 累加
                    float sum = 0.0f;
                    for (size_t c = 0; c < C; ++c) {
                        for (size_t kh = 0; kh < KH; ++kh) {
                            for (size_t kw = 0; kw < KW; ++kw) {
                                int in_h = h_start + kh * dilation;
                                int in_w = w_start + kw * dilation;

                                // 检查边界（padding 区域为 0）
                                if (in_h < 0 || in_h >= (int)H ||
                                    in_w < 0 || in_w >= (int)W) {
                                    continue;
                                }

                                float inp = Conv::to_float(
                                    input.data[n * C * H * W +
                                               c * H * W +
                                               in_h * W +
                                               in_w]
                                );
                                float wgt = Conv::to_float(
                                    weight.data[o * C * KH * KW +
                                                c * KH * KW +
                                                kh * KW +
                                                kw]
                                );
                                sum += inp * wgt;
                            }
                        }
                    }

                    // 加上 bias
                    if (bias != nullptr) {
                        sum += Conv::to_float(bias->data[o]);
                    }

                    // 存入输出
                    output.data[n * O * H_out * W_out +
                                o * H_out * W_out +
                                h * W_out +
                                w] = Conv::from_float(sum);
                }
            }
        }
    }

    return output;
}

} // namespace infer_train
