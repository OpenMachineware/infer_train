#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"

namespace infer_train {

// ============================================================
// cumsum: 沿指定维度累加
// ============================================================
template<typename T>
Tensor<T> cumsum(const Tensor<T>& input, int dim = -1) {
    if (input.shape.empty()) return Tensor<T>();

    int ndim = static_cast<int>(input.shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return Tensor<T>();

    using Conv = DTypeConverter<T>;
    Tensor<T> output(input.shape);

    // 计算步长
    std::vector<size_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * input.shape[i + 1];
    }

    size_t dim_stride = strides[dim];
    size_t dim_size = input.shape[dim];
    size_t outer_size = input.size() / (dim_size * dim_stride);

    for (size_t outer = 0; outer < outer_size; ++outer) {
        size_t base = outer * dim_size * dim_stride;
        float running = 0.0f;
        for (size_t d = 0; d < dim_size; ++d) {
            size_t idx = base + d * dim_stride;
            for (size_t s = 0; s < dim_stride; ++s) {
                size_t pos = idx + s;
                running += Conv::to_float(input.data[pos]);
                output.data[pos] = Conv::from_float(running);
            }
        }
    }

    return output;
}

} // namespace infer_train
