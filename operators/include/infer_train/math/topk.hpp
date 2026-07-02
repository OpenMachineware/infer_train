#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <vector>
#include <algorithm>
#include <queue>

namespace infer_train {

// ============================================================
// topk: 沿指定维度取前 k 大值
// ============================================================
template<typename T>
std::pair<Tensor<T>, std::vector<int64_t>> topk(
    const Tensor<T>& input,
    size_t k,
    int dim = -1,
    bool largest = true,
    bool sorted = true
) {
    if (input.shape.empty()) {
        return {Tensor<T>(), std::vector<int64_t>()};
    }

    int ndim = static_cast<int>(input.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        return {Tensor<T>(), std::vector<int64_t>()};
    }

    size_t dim_size = input.shape[dim];
    if (k > dim_size) k = dim_size;
    if (k == 0) {
        return {Tensor<T>(), std::vector<int64_t>()};
    }

    // 输出形状
    std::vector<size_t> out_shape = input.shape;
    out_shape[dim] = k;

    Tensor<T> values(out_shape);
    std::vector<int64_t> indices(input.size(), 0);

    // 计算步长
    std::vector<size_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * input.shape[i + 1];
    }

    size_t dim_stride = strides[dim];
    size_t outer_size = input.size() / (dim_size * dim_stride);

    using Conv = DTypeConverter<T>;

    for (size_t outer = 0; outer < outer_size; ++outer) {
        size_t base = outer * dim_size * dim_stride;

        // 收集当前切片的 (value, index)
        std::vector<std::pair<float, size_t>> items;
        for (size_t d = 0; d < dim_size; ++d) {
            size_t pos = base + d * dim_stride;
            for (size_t s = 0; s < dim_stride; ++s) {
                float val = Conv::to_float(input.data[pos + s]);
                items.push_back({val, d});
            }
        }

        // 排序
        if (largest) {
            std::sort(items.begin(), items.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
        } else {
            std::sort(items.begin(), items.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        // 取前 k 个
        for (size_t i = 0; i < k; ++i) {
            size_t out_pos = base + i * dim_stride;
            for (size_t s = 0; s < dim_stride; ++s) {
                values.data[out_pos + s] = Conv::from_float(items[i].first);
                indices[out_pos + s] = static_cast<int64_t>(items[i].second);
            }
        }
    }

    return {values, indices};
}

} // namespace infer_train
