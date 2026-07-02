#pragma once
#include "infer_train/quantized/core.hpp"
#include "infer_train/nn/attention.hpp"

namespace infer_train {

template<typename T>
Tensor<T> quantized_scaled_dot_product_attention(
    const Tensor<T>& query,
    const Tensor<T>& key,
    const Tensor<T>& value,
    const Tensor<T>* mask = nullptr,
    float scale = -1.0f,
    bool is_causal = false,
    float dropout_p = 0.0f
) {
    static_assert(is_quantized<T>::value,
                  "quantized_scaled_dot_product_attention only works with quantized types");

    Tensor<F32> q_fp32 = dequantize(query);
    Tensor<F32> k_fp32 = dequantize(key);
    Tensor<F32> v_fp32 = dequantize(value);

    Tensor<F32> mask_fp32;
    bool has_mask = (mask != nullptr);
    if (has_mask) {
        mask_fp32 = dequantize(*mask);
    }

    Tensor<F32> result_fp32 = scaled_dot_product_attention<F32>(
        q_fp32, k_fp32, v_fp32,
        has_mask ? &mask_fp32 : nullptr,
        scale, is_causal, dropout_p
    );

    float out_scale, out_zero_point;
    compute_scale_zero_point(result_fp32, out_scale, out_zero_point);
    return quantize<T>(result_fp32, out_scale, out_zero_point);
}

template<typename T>
Tensor<T> quantized_multi_head_attention(
    const Tensor<T>& query,
    const Tensor<T>& key,
    const Tensor<T>& value,
    const Tensor<T>* mask = nullptr,
    int num_heads = 8,
    float scale = -1.0f,
    bool is_causal = false,
    float dropout_p = 0.0f
) {
    static_assert(is_quantized<T>::value,
                  "quantized_multi_head_attention only works with quantized types");

    // 简化：只支持单头
    if (num_heads != 1) {
        return Tensor<T>();  // 暂不支持多头量化
    }

    return quantized_scaled_dot_product_attention(
        query, key, value, mask, scale, is_causal, dropout_p
    );
}

} // namespace infer_train
