#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>
#include <cmath>

namespace infer_train {

// ============================================================
// slice: 沿指定维度切取（支持任意维度）
// ============================================================
template<typename T>
Tensor<T> slice(
    const Tensor<T>& input,
    int dim,
    int start,
    int end,
    int step = 1
) {
    int ndim = static_cast<int>(input.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return Tensor<T>();

    if (start < 0) start += static_cast<int>(input.shape[dim]);
    if (end < 0) end += static_cast<int>(input.shape[dim]);
    if (start < 0) start = 0;
    if (end > static_cast<int>(input.shape[dim])) end = static_cast<int>(input.shape[dim]);
    if (start >= end) return Tensor<T>();

    // 计算输出形状
    int slice_len = 0;
    for (int i = start; i < end; i += step) slice_len++;
    if (slice_len <= 0) return Tensor<T>();

    std::vector<size_t> out_shape = input.shape;
    out_shape[dim] = static_cast<size_t>(slice_len);

    Tensor<T> output(out_shape);

    // 计算输入步长
    std::vector<size_t> in_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * input.shape[i + 1];
    }

    // 计算输出步长
    std::vector<size_t> out_strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // 遍历输出所有元素
    std::vector<size_t> idx(ndim, 0);
    for (size_t i = 0; i < output.size(); ++i) {
        // 计算对应的输入索引
        size_t input_idx = 0;
        size_t offset = i;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t pos = offset / out_strides[d];
            offset %= out_strides[d];
            if (d == dim) {
                pos = start + pos * step;
            }
            input_idx += pos * in_strides[d];
        }
        output.data[i] = input.data[input_idx];
    }

    return output;
}

} // namespace infer_train
