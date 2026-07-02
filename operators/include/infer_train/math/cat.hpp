#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>

namespace infer_train {

// ============================================================
// cat: 沿指定维度拼接多个张量
// ============================================================
template<typename T>
Tensor<T> cat(const std::vector<const Tensor<T>*>& tensors, int dim) {
    if (tensors.empty()) return Tensor<T>();

    int ndim = static_cast<int>(tensors[0]->shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return Tensor<T>();

    // 检查所有张量形状是否兼容
    std::vector<size_t> out_shape = tensors[0]->shape;
    size_t cat_dim_size = 0;
    for (const auto* t : tensors) {
        if (t->shape.size() != static_cast<size_t>(ndim)) return Tensor<T>();
        for (int d = 0; d < ndim; ++d) {
            if (d == dim) continue;
            if (t->shape[d] != out_shape[d]) return Tensor<T>();
        }
        cat_dim_size += t->shape[dim];
    }
    out_shape[dim] = cat_dim_size;

    Tensor<T> output(out_shape);

    // 计算步长
    std::vector<size_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * out_shape[i + 1];
    }

    size_t offset = 0;
    for (const auto* t : tensors) {
        size_t t_size = t->size();
        // 将 t 的数据复制到 output 的对应位置
        // 需要按维度索引复制，不能用 memcpy（因为有步长）
        // 简化：用循环逐元素复制
        for (size_t i = 0; i < t_size; ++i) {
            // 计算在 output 中的位置
            size_t out_idx = 0;
            size_t tmp = i;
            size_t stride_factor = 1;
            for (int d = ndim - 1; d >= 0; --d) {
                size_t pos = tmp % t->shape[d];
                tmp /= t->shape[d];
                size_t out_pos = pos;
                if (d == dim) {
                    out_pos += offset;
                }
                out_idx += out_pos * strides[d];
            }
            output.data[out_idx] = t->data[i];
        }
        offset += t->shape[dim];
    }

    return output;
}

} // namespace infer_train
