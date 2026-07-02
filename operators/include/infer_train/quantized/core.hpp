#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <algorithm>
#include <cmath>

namespace infer_train {

// ============================================================
// dequantize
// ============================================================
template<typename T>
Tensor<F32> dequantize(const Tensor<T>& input) {
    using Conv = QuantizedConverter<T>;
    Tensor<F32> result(input.shape);

    for (size_t i = 0; i < result.size(); ++i) {
        result.data[i] = Conv::dequantize(
            input.data[i],
            input.scale,
            input.zero_point
        );
    }
    return result;
}

// ============================================================
// quantize
// ============================================================
template<typename T>
Tensor<T> quantize(const Tensor<F32>& input, float scale, float zero_point) {
    using Conv = QuantizedConverter<T>;
    Tensor<T> result(input.shape);
    result.scale = scale;
    result.zero_point = zero_point;

    for (size_t i = 0; i < result.size(); ++i) {
        result.data[i] = Conv::quantize(input.data[i], scale, zero_point);
    }
    return result;
}

// ============================================================
// compute_scale_zero_point
// ============================================================
inline void compute_scale_zero_point(
    const Tensor<F32>& input,
    float& scale,
    float& zero_point
) {
    float min_val = input.data[0];
    float max_val = input.data[0];
    for (size_t i = 1; i < input.size(); ++i) {
        min_val = std::min(min_val, input.data[i]);
        max_val = std::max(max_val, input.data[i]);
    }

    scale = (max_val - min_val) / 255.0f;
    if (scale < 1e-7f) scale = 1e-7f;
    zero_point = -min_val / scale - 128.0f;
    zero_point = std::max(-128.0f, std::min(127.0f, zero_point));
}

} // namespace infer_train
