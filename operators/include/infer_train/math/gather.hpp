#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>

namespace infer_train {

// ============================================================
// gather: 沿指定维度按索引取值
// input: (N, ...)
// indices: (N, ..., K) 索引
// output: (N, ..., K)
// ============================================================
template<typename T>
Tensor<T> gather(
    const Tensor<T>& input,
    const std::vector<int64_t>& indices,
    const std::vector<size_t>& indices_shape,
    int dim = 0
) {
    if (input.shape.size() != indices_shape.size()) return Tensor<T>();

    int ndim = static_cast<int>(input.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return Tensor<T>();

    // 检查形状：除了 dim 维度，其他维度必须一致
    for (int d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        if (input.shape[d] != indices_shape[d]) return Tensor<T>();
    }

    Tensor<T> output(indices_shape);

    // 计算步长
    std::vector<size_t> in_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input.shape[i + 1];
    }

    std::vector<size_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * indices_shape[i + 1];
    }

    // 遍历输出
    for (size_t i = 0; i < output.size(); ++i) {
        size_t idx = i;
        size_t input_idx = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t pos = idx / out_strides[d];
            idx %= out_strides[d];
            if (d == dim) {
                int64_t idx_val = indices[i];
                if (idx_val < 0) idx_val += input.shape[d];
                if (idx_val < 0 || idx_val >= (int64_t)input.shape[d]) {
                    return Tensor<T>();
                }
                input_idx += (size_t)idx_val * in_strides[d];
            } else {
                input_idx += pos * in_strides[d];
            }
        }
        output.data[i] = input.data[input_idx];
    }

    return output;
}

} // namespace infer_train
