#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"

namespace infer_train {

// ============================================================
// 浮点矩阵乘法（模板统一处理）
// ============================================================
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_matmul for quantized types");

    if (a.shape.size() != 2 || b.shape.size() != 2) {
        return Tensor<T>();
    }

    if (a.shape[1] != b.shape[0]) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    Tensor<T> result({M, N});

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float fa = Conv::to_float(a.data[i * K + k]);
                float fb = Conv::to_float(b.data[k * N + j]);
                sum += fa * fb;
            }
            result.data[i * N + j] = Conv::from_float(sum);
        }
    }

    return result;
}

// ============================================================
// 浮点批量矩阵乘法
// ============================================================
template<typename T>
Tensor<T> batch_matmul(const Tensor<T>& a, const Tensor<T>& b) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_batch_matmul for quantized types");

    if (a.shape.size() != 3 || b.shape.size() != 3) {
        return Tensor<T>();
    }

    if (a.shape[0] != b.shape[0] || a.shape[2] != b.shape[1]) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t batch = a.shape[0];
    size_t M = a.shape[1];
    size_t K = a.shape[2];
    size_t N = b.shape[2];

    Tensor<T> result({batch, M, N});

    for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
        const auto* a_start = &a.data[b_idx * M * K];
        const auto* b_start = &b.data[b_idx * K * N];
        auto* c_start = &result.data[b_idx * M * N];

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    float fa = Conv::to_float(a_start[i * K + k]);
                    float fb = Conv::to_float(b_start[k * N + j]);
                    sum += fa * fb;
                }
                c_start[i * N + j] = Conv::from_float(sum);
            }
        }
    }

    return result;
}

// ============================================================
// 浮点转置
// ============================================================
template<typename T>
Tensor<T> transpose(const Tensor<T>& a) {
    if (a.shape.size() != 2) {
        return Tensor<T>();
    }

    size_t rows = a.shape[0];
    size_t cols = a.shape[1];

    Tensor<T> result({cols, rows});

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j * rows + i] = a.data[i * cols + j];
        }
    }

    return result;
}

// ============================================================
// 向量 × 矩阵：1D * 2D = 1D
// ============================================================
template<typename T>
Tensor<T> vec_matmul(const Tensor<T>& vec, const Tensor<T>& mat) {
    static_assert(!is_quantized<T>::value,
                  "Use quantized_vec_matmul for quantized types");

    if (vec.shape.size() != 1 || mat.shape.size() != 2) {
        return Tensor<T>();
    }

    if (vec.shape[0] != mat.shape[0]) {
        return Tensor<T>();
    }

    using Conv = DTypeConverter<T>;

    size_t K = vec.shape[0];
    size_t N = mat.shape[1];

    Tensor<T> result({N});

    for (size_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            float fv = Conv::to_float(vec.data[k]);
            float fm = Conv::to_float(mat.data[k * N + j]);
            sum += fv * fm;
        }
        result.data[j] = Conv::from_float(sum);
    }

    return result;
}

} // namespace infer_train









