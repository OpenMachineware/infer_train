#include "c_api_common.h"
#include "infer_train/nn.hpp"

using namespace infer_train;

// ============================================================
// scaled_dot_product_attention
// ============================================================
extern "C" it_tensor* it_scaled_dot_product_attention(
    const it_tensor* query,
    const it_tensor* key,
    const it_tensor* value,
    const it_tensor* mask,
    float scale,
    int is_causal,
    float dropout_p
) {
    if (query->dtype != key->dtype || query->dtype != value->dtype) {
        return nullptr;
    }

    switch (query->dtype) {
        case IT_DTYPE_F32: {
            auto q = to_cpp_tensor<F32>(query);
            auto k = to_cpp_tensor<F32>(key);
            auto v = to_cpp_tensor<F32>(value);
            const Tensor<F32>* mask_ptr = nullptr;
            Tensor<F32> mask_tensor;
            if (mask) { mask_tensor = to_cpp_tensor<F32>(mask); mask_ptr = &mask_tensor; }
            auto result = scaled_dot_product_attention(q, k, v, mask_ptr, scale, is_causal != 0, dropout_p);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto q = to_cpp_tensor<F64>(query);
            auto k = to_cpp_tensor<F64>(key);
            auto v = to_cpp_tensor<F64>(value);
            const Tensor<F64>* mask_ptr = nullptr;
            Tensor<F64> mask_tensor;
            if (mask) { mask_tensor = to_cpp_tensor<F64>(mask); mask_ptr = &mask_tensor; }
            auto result = scaled_dot_product_attention(q, k, v, mask_ptr, scale, is_causal != 0, dropout_p);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto q = to_cpp_tensor<F16>(query);
            auto k = to_cpp_tensor<F16>(key);
            auto v = to_cpp_tensor<F16>(value);
            const Tensor<F16>* mask_ptr = nullptr;
            Tensor<F16> mask_tensor;
            if (mask) { mask_tensor = to_cpp_tensor<F16>(mask); mask_ptr = &mask_tensor; }
            auto result = scaled_dot_product_attention(q, k, v, mask_ptr, scale, is_causal != 0, dropout_p);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto q = to_cpp_tensor<BF16>(query);
            auto k = to_cpp_tensor<BF16>(key);
            auto v = to_cpp_tensor<BF16>(value);
            const Tensor<BF16>* mask_ptr = nullptr;
            Tensor<BF16> mask_tensor;
            if (mask) { mask_tensor = to_cpp_tensor<BF16>(mask); mask_ptr = &mask_tensor; }
            auto result = scaled_dot_product_attention(q, k, v, mask_ptr, scale, is_causal != 0, dropout_p);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// multi_head_attention
// ============================================================
extern "C" it_tensor* it_multi_head_attention(
    const it_tensor* query,
    const it_tensor* key,
    const it_tensor* value,
    const it_tensor* mask,
    int num_heads,
    float scale,
    int is_causal,
    float dropout_p
) {
    if (query->dtype != key->dtype || query->dtype != value->dtype) {
        return nullptr;
    }

    switch (query->dtype) {
        case IT_DTYPE_F32: {
            auto q = to_cpp_tensor<F32>(query);
            auto k = to_cpp_tensor<F32>(key);
            auto v = to_cpp_tensor<F32>(value);
            const Tensor<F32>* mask_ptr = nullptr;
            Tensor<F32> mask_tensor;
            if (mask) { mask_tensor = to_cpp_tensor<F32>(mask); mask_ptr = &mask_tensor; }
            auto result = multi_head_attention(q, k, v, mask_ptr, num_heads, scale, is_causal != 0, dropout_p);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto q = to_cpp_tensor<F64>(query);
            auto k = to_cpp_tensor<F64>(key);
            auto v = to_cpp_tensor<F64>(value);
            const Tensor<F64>* mask_ptr = nullptr;
            Tensor<F64> mask_tensor;
            if (mask) { mask_tensor = to_cpp_tensor<F64>(mask); mask_ptr = &mask_tensor; }
            auto result = multi_head_attention(q, k, v, mask_ptr, num_heads, scale, is_causal != 0, dropout_p);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto q = to_cpp_tensor<F16>(query);
            auto k = to_cpp_tensor<F16>(key);
            auto v = to_cpp_tensor<F16>(value);
            const Tensor<F16>* mask_ptr = nullptr;
            Tensor<F16> mask_tensor;
            if (mask) { mask_tensor = to_cpp_tensor<F16>(mask); mask_ptr = &mask_tensor; }
            auto result = multi_head_attention(q, k, v, mask_ptr, num_heads, scale, is_causal != 0, dropout_p);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto q = to_cpp_tensor<BF16>(query);
            auto k = to_cpp_tensor<BF16>(key);
            auto v = to_cpp_tensor<BF16>(value);
            const Tensor<BF16>* mask_ptr = nullptr;
            Tensor<BF16> mask_tensor;
            if (mask) { mask_tensor = to_cpp_tensor<BF16>(mask); mask_ptr = &mask_tensor; }
            auto result = multi_head_attention(q, k, v, mask_ptr, num_heads, scale, is_causal != 0, dropout_p);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// rotary_embedding
// ============================================================
extern "C" it_tensor* it_rotary_embedding(
    const it_tensor* x,
    const it_tensor* cos,
    const it_tensor* sin
) {
    if (x->dtype != cos->dtype || x->dtype != sin->dtype) return nullptr;

    switch (x->dtype) {
        case IT_DTYPE_F32: {
            auto tx = to_cpp_tensor<F32>(x);
            auto tc = to_cpp_tensor<F32>(cos);
            auto ts = to_cpp_tensor<F32>(sin);
            auto result = rotary_embedding(tx, tc, ts);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto tx = to_cpp_tensor<F64>(x);
            auto tc = to_cpp_tensor<F64>(cos);
            auto ts = to_cpp_tensor<F64>(sin);
            auto result = rotary_embedding(tx, tc, ts);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto tx = to_cpp_tensor<F16>(x);
            auto tc = to_cpp_tensor<F16>(cos);
            auto ts = to_cpp_tensor<F16>(sin);
            auto result = rotary_embedding(tx, tc, ts);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto tx = to_cpp_tensor<BF16>(x);
            auto tc = to_cpp_tensor<BF16>(cos);
            auto ts = to_cpp_tensor<BF16>(sin);
            auto result = rotary_embedding(tx, tc, ts);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}
