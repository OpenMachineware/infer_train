#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/embedding.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_embedding(
    const std::vector<int64_t>& indices,
    const Tensor<T>& weight,
    int padding_idx = -1
) {
    static_assert(is_quantized<T>::value,
                  "quantized_embedding only works with quantized types");
    
    Tensor<F32> fp32_weight = dequantize(weight);
    Tensor<F32> result_fp32 = embedding<F32>(indices, fp32_weight, padding_idx);
    
    float scale, zero_point;
    compute_scale_zero_point(result_fp32, scale, zero_point);
    return quantize<T>(result_fp32, scale, zero_point);
}

} // namespace infer_train
