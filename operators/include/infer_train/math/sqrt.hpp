#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

template<typename T>
Tensor<T> sqrt(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::sqrt(std::max(0.0f, x)));
    }

    return result;
}

template<typename T>
Tensor<T> rsqrt(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(1.0f / std::sqrt(std::max(0.0f, x)));
    }

    return result;
}

// ============================================================
// I8 专用版本（直接比较存储值）
// ============================================================
inline std::pair<Tensor<I8>, std::vector<int64_t>> sort(
    const Tensor<I8>& input,
    int dim = -1,
    bool ascending = true
) {
    if (input.shape.empty()) {
        return {Tensor<I8>(), std::vector<int64_t>()};
    }

    int ndim = static_cast<int>(input.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        return {Tensor<I8>(), std::vector<int64_t>()};
    }

    Tensor<I8> values(input.shape);
    std::vector<int64_t> indices(input.size(), 0);

    std::vector<size_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * input.shape[i + 1];
    }

    size_t dim_stride = strides[dim];
    size_t dim_size = input.shape[dim];
    size_t outer_size = input.size() / (dim_size * dim_stride);

    for (size_t outer = 0; outer < outer_size; ++outer) {
        size_t base = outer * dim_size * dim_stride;

        std::vector<std::pair<int8_t, size_t>> items;
        for (size_t d = 0; d < dim_size; ++d) {
            size_t pos = base + d * dim_stride;
            for (size_t s = 0; s < dim_stride; ++s) {
                items.push_back({input.data[pos + s], d});
            }
        }

        if (ascending) {
            std::sort(items.begin(), items.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        } else {
            std::sort(items.begin(), items.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
        }

        for (size_t i = 0; i < dim_size; ++i) {
            size_t pos = base + i * dim_stride;
            for (size_t s = 0; s < dim_stride; ++s) {
                values.data[pos + s] = items[i].first;
                indices[pos + s] = static_cast<int64_t>(items[i].second);
            }
        }
    }

    return {values, indices};
}

} // namespace infer_train
