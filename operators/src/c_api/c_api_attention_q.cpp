#include "c_api_common.h"
#include "infer_train/nn_q.hpp"

using namespace infer_train;

extern "C" it_tensor* it_quantized_scaled_dot_product_attention(
    const it_tensor* query,
    const it_tensor* key,
    const it_tensor* value,
    const it_tensor* mask,
    float scale,
    int is_causal,
    float dropout_p
) {
    if (query->dtype != IT_DTYPE_I8 || key->dtype != IT_DTYPE_I8 || value->dtype != IT_DTYPE_I8) {
        return nullptr;
    }

    const Tensor<I8>* mask_ptr = nullptr;
    Tensor<I8> mask_tensor;
    bool has_mask = (mask != nullptr);
    if (has_mask) {
        mask_tensor = to_cpp_tensor_with_params<I8>(mask);
        mask_ptr = &mask_tensor;
    }

    auto q = to_cpp_tensor_with_params<I8>(query);
    auto k = to_cpp_tensor_with_params<I8>(key);
    auto v = to_cpp_tensor_with_params<I8>(value);
    auto result = quantized_scaled_dot_product_attention(q, k, v, mask_ptr, scale, is_causal != 0, dropout_p);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_multi_head_attention(
    const it_tensor* query,
    const it_tensor* key,
    const it_tensor* value,
    const it_tensor* mask,
    int num_heads,
    float scale,
    int is_causal,
    float dropout_p
) {
    if (query->dtype != IT_DTYPE_I8 || key->dtype != IT_DTYPE_I8 || value->dtype != IT_DTYPE_I8) {
        return nullptr;
    }

    const Tensor<I8>* mask_ptr = nullptr;
    Tensor<I8> mask_tensor;
    bool has_mask = (mask != nullptr);
    if (has_mask) {
        mask_tensor = to_cpp_tensor_with_params<I8>(mask);
        mask_ptr = &mask_tensor;
    }

    auto q = to_cpp_tensor_with_params<I8>(query);
    auto k = to_cpp_tensor_with_params<I8>(key);
    auto v = to_cpp_tensor_with_params<I8>(value);
    auto result = quantized_multi_head_attention(q, k, v, mask_ptr, num_heads, scale, is_causal != 0, dropout_p);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}

extern "C" it_tensor* it_quantized_rotary_embedding(
    const it_tensor* x,
    const it_tensor* cos,
    const it_tensor* sin
) {
    if (x->dtype != IT_DTYPE_I8 || cos->dtype != IT_DTYPE_I8 || sin->dtype != IT_DTYPE_I8) {
        return nullptr;
    }

    auto tx = to_cpp_tensor_with_params<I8>(x);
    auto tc = to_cpp_tensor_with_params<I8>(cos);
    auto ts = to_cpp_tensor_with_params<I8>(sin);
    auto result = quantized_rotary_embedding(tx, tc, ts);
    return from_cpp_tensor(result, IT_DTYPE_I8);
}
