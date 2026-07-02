#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/math/matmul.hpp"
#include <vector>

namespace infer_train {

// ============================================================
// quantized_matmul（INT32 累加）
// ============================================================
template<typename T>
Tensor<T> quantized_matmul(const Tensor<T>& a, const Tensor<T>& b) {
    static_assert(is_quantized<T>::value,
                  "quantized_matmul only works with quantized types");

    if (a.shape.size() != 2 || b.shape.size() != 2) {
        return Tensor<T>();
    }

    if (a.shape[1] != b.shape[0]) {
        return Tensor<T>();
    }

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    // INT32 累加
    std::vector<int32_t> acc(M * N, 0);
    int32_t a_zp = static_cast<int32_t>(a.zero_point);
    int32_t b_zp = static_cast<int32_t>(b.zero_point);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                int32_t ia = static_cast<int32_t>(a.data[i * K + k]) - a_zp;
                int32_t ib = static_cast<int32_t>(b.data[k * N + j]) - b_zp;
                sum += ia * ib;
            }
            acc[i * N + j] = sum;
        }
    }

    // 转 FP32
    Tensor<F32> result_fp32({M, N});
    float scale_factor = a.scale * b.scale;
    for (size_t i = 0; i < result_fp32.size(); ++i) {
        result_fp32.data[i] = static_cast<float>(acc[i]) * scale_factor;
    }

    float out_scale, out_zero_point;
    compute_scale_zero_point(result_fp32, out_scale, out_zero_point);
    return quantize<T>(result_fp32, out_scale, out_zero_point);
}

// ============================================================
// quantized_vec_matmul
// ============================================================
template<typename T>
Tensor<T> quantized_vec_matmul(const Tensor<T>& vec, const Tensor<T>& mat) {
    static_assert(is_quantized<T>::value,
                  "quantized_vec_matmul only works with quantized types");

    if (vec.shape.size() != 1 || mat.shape.size() != 2) {
        return Tensor<T>();
    }

    if (vec.shape[0] != mat.shape[0]) {
        return Tensor<T>();
    }

    size_t K = vec.shape[0];
    size_t N = mat.shape[1];

    std::vector<int32_t> acc(N, 0);
    int32_t vec_zp = static_cast<int32_t>(vec.zero_point);
    int32_t mat_zp = static_cast<int32_t>(mat.zero_point);

    for (size_t k = 0; k < K; ++k) {
        int32_t iv = static_cast<int32_t>(vec.data[k]) - vec_zp;
        for (size_t j = 0; j < N; ++j) {
            int32_t im = static_cast<int32_t>(mat.data[k * N + j]) - mat_zp;
            acc[j] += iv * im;
        }
    }

    Tensor<F32> result_fp32({N});
    float scale_factor = vec.scale * mat.scale;
    for (size_t j = 0; j < N; ++j) {
        result_fp32.data[j] = static_cast<float>(acc[j]) * scale_factor;
    }

    float out_scale, out_zero_point;
    compute_scale_zero_point(result_fp32, out_scale, out_zero_point);
    return quantize<T>(result_fp32, out_scale, out_zero_point);
}

// ============================================================
// quantized_transpose
// ============================================================
template<typename T>
Tensor<T> quantized_transpose(const Tensor<T>& input) {
    static_assert(is_quantized<T>::value,
                  "quantized_transpose only works with quantized types");

    Tensor<F32> fp32_input = dequantize(input);
    Tensor<F32> result_fp32 = transpose<F32>(fp32_input);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train
