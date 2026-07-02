#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>

namespace infer_train {

// ============================================================
// scatter: 沿指定维度按索引写入
// input: (N, ...)
// indices: (N, ..., K) 索引
// value: 标量或同形状张量
// ============================================================
template<typename T>
Tensor<T> scatter(
    const Tensor<T>& input,
    const std::vector<int64_t>& indices,
    const std::vector<size_t>& indices_shape,
    typename T::storage value,
    int dim = 0
) {
    if (input.shape.size() != indices_shape.size()) return Tensor<T>();

    int ndim = static_cast<int>(input.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return Tensor<T>();

    Tensor<T> output = input;  // 复制

    // 计算步长
    std::vector<size_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * input.shape[i + 1];
    }

    std::vector<size_t> idx_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        idx_strides[i] = idx_strides[i + 1] * indices_shape[i + 1];
    }

    using Conv = DTypeConverter<T>;
    float val = Conv::to_float(value);

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = i;
        size_t out_idx = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t pos = idx / idx_strides[d];
            idx %= idx_strides[d];
            if (d == dim) {
                int64_t idx_val = indices[i];
                if (idx_val < 0) idx_val += input.shape[d];
                if (idx_val < 0 || idx_val >= (int64_t)input.shape[d]) {
                    return Tensor<T>();
                }
                out_idx += (size_t)idx_val * strides[d];
            } else {
                // 对于非 dim 维度，pos 应该和 input 对应位置一致
                // 但 scatter 的 indices 形状在 dim 维度是 K
                // 其他维度和 input 一致
                out_idx += pos * strides[d];
            }
        }
        output.data[out_idx] = Conv::from_float(val);
    }

    return output;
}

// 张量版本：用 src 填充
template<typename T>
Tensor<T> scatter(
    const Tensor<T>& input,
    const std::vector<int64_t>& indices,
    const std::vector<size_t>& indices_shape,
    const Tensor<T>& src,
    int dim = 0
) {
    if (input.shape.size() != indices_shape.size()) return Tensor<T>();
    if (indices_shape != src.shape) return Tensor<T>();

    int ndim = static_cast<int>(input.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return Tensor<T>();

    Tensor<T> output = input;

    std::vector<size_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * input.shape[i + 1];
    }

    std::vector<size_t> idx_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        idx_strides[i] = idx_strides[i + 1] * indices_shape[i + 1];
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = i;
        size_t out_idx = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t pos = idx / idx_strides[d];
            idx %= idx_strides[d];
            if (d == dim) {
                int64_t idx_val = indices[i];
                if (idx_val < 0) idx_val += input.shape[d];
                if (idx_val < 0 || idx_val >= (int64_t)input.shape[d]) {
                    return Tensor<T>();
                }
                out_idx += (size_t)idx_val * strides[d];
            } else {
                out_idx += pos * strides[d];
            }
        }
        output.data[out_idx] = src.data[i];
    }

    return output;
}

} // namespace infer_train
