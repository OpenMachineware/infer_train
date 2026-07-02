#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/rotary_embedding.hpp"

namespace infer_train {

// ============================================================
// 量化 Rotary Embedding
// ============================================================
template<typename T>
Tensor<T> quantized_rotary_embedding(
    const Tensor<T>& x,
    const Tensor<T>& cos,
    const Tensor<T>& sin
) {
    static_assert(is_quantized<T>::value,
                  "quantized_rotary_embedding only works with quantized types");

    Tensor<F32> x_fp32 = dequantize(x);
    Tensor<F32> cos_fp32 = dequantize(cos);
    Tensor<F32> sin_fp32 = dequantize(sin);

    Tensor<F32> result_fp32 = rotary_embedding<F32>(x_fp32, cos_fp32, sin_fp32);

    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train
