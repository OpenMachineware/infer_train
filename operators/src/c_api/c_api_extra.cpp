#include "c_api_common.h"
#include "infer_train/math.hpp"
#include "infer_train/math/cat.hpp"
#include "infer_train/nn.hpp"
#include <vector>

using namespace infer_train;

// ============================================================
// slice
// ============================================================
extern "C" it_tensor* it_slice(
    const it_tensor* input,
    int dim,
    int start,
    int end,
    int step
) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = slice(t, dim, start, end, step);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = slice(t, dim, start, end, step);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = slice(t, dim, start, end, step);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = slice(t, dim, start, end, step);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        case IT_DTYPE_I8: {
            auto t = to_cpp_tensor<I8>(input);
            auto result = slice(t, dim, start, end, step);
            return from_cpp_tensor(result, IT_DTYPE_I8);
        }
        default: return nullptr;
    }
}

// ============================================================
// cat
// ============================================================
extern "C" it_tensor* it_cat(const it_tensor** tensors, size_t n, int dim) {
    if (n == 0 || tensors == nullptr) return nullptr;

    it_dtype_t dtype = tensors[0]->dtype;
    for (size_t i = 1; i < n; ++i) {
        if (tensors[i]->dtype != dtype) return nullptr;
    }

    switch (dtype) {
        case IT_DTYPE_F32: {
            std::vector<const Tensor<F32>*> vec;
            std::vector<Tensor<F32>> storage;
            storage.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                storage.push_back(to_cpp_tensor<F32>(tensors[i]));
                vec.push_back(&storage.back());
            }
            auto result = cat(vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            std::vector<const Tensor<F64>*> vec;
            std::vector<Tensor<F64>> storage;
            storage.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                storage.push_back(to_cpp_tensor<F64>(tensors[i]));
                vec.push_back(&storage.back());
            }
            auto result = cat(vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            std::vector<const Tensor<F16>*> vec;
            std::vector<Tensor<F16>> storage;
            storage.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                storage.push_back(to_cpp_tensor<F16>(tensors[i]));
                vec.push_back(&storage.back());
            }
            auto result = cat(vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            std::vector<const Tensor<BF16>*> vec;
            std::vector<Tensor<BF16>> storage;
            storage.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                storage.push_back(to_cpp_tensor<BF16>(tensors[i]));
                vec.push_back(&storage.back());
            }
            auto result = cat(vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        case IT_DTYPE_I8: {
            std::vector<const Tensor<I8>*> vec;
            std::vector<Tensor<I8>> storage;
            storage.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                storage.push_back(to_cpp_tensor<I8>(tensors[i]));
                vec.push_back(&storage.back());
            }
            auto result = cat(vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_I8);
        }
        default: return nullptr;
    }
}

// ============================================================
// cumsum
// ============================================================
extern "C" it_tensor* it_cumsum(const it_tensor* input, int dim) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = cumsum(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = cumsum(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = cumsum(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = cumsum(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// cumprod
// ============================================================
extern "C" it_tensor* it_cumprod(const it_tensor* input, int dim) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = cumprod(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = cumprod(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = cumprod(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = cumprod(t, dim);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// where
// ============================================================
extern "C" it_tensor* it_where(
    const uint8_t* condition,
    const size_t* condition_shape,
    size_t condition_ndim,
    const it_tensor* true_val,
    const it_tensor* false_val
) {
    if (true_val->dtype != false_val->dtype) return nullptr;

    std::vector<uint8_t> cond(condition, condition + it_tensor_size(true_val));
    std::vector<size_t> shape(condition_shape, condition_shape + condition_ndim);

    switch (true_val->dtype) {
        case IT_DTYPE_F32: {
            auto tv = to_cpp_tensor<F32>(true_val);
            auto fv = to_cpp_tensor<F32>(false_val);
            auto result = where(cond, shape, tv, fv);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto tv = to_cpp_tensor<F64>(true_val);
            auto fv = to_cpp_tensor<F64>(false_val);
            auto result = where(cond, shape, tv, fv);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto tv = to_cpp_tensor<F16>(true_val);
            auto fv = to_cpp_tensor<F16>(false_val);
            auto result = where(cond, shape, tv, fv);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto tv = to_cpp_tensor<BF16>(true_val);
            auto fv = to_cpp_tensor<BF16>(false_val);
            auto result = where(cond, shape, tv, fv);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// argmax
// ============================================================
extern "C" it_tensor* it_argmax(const it_tensor* input) {
    // 检查输入是否有效
    if (!input) return nullptr;
    if (input->ndim == 0) return nullptr;

    // 将 C 张量转换为 C++ 张量
    // 返回的 it_tensor 需要包含 int64_t 数据

    // 遍历所有元素，找最大值
    // 根据类型分派
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = argmax(t);
            // result 是 std::vector<int64_t>
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
            ct->ndim = 1;
            ct->shape = (size_t*)malloc(sizeof(size_t));
            ct->shape[0] = result.size();
            ct->data = malloc(result.size() * sizeof(int64_t));
            memcpy(ct->data, result.data(), result.size() * sizeof(int64_t));
            ct->elem_size = sizeof(int64_t);
            ct->dtype = IT_DTYPE_I64;
            ct->scale = 1.0f;
            ct->zero_point = 0.0f;
            ct->owns_memory = true;
            return ct;
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = argmax(t);
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
            ct->ndim = 1;
            ct->shape = (size_t*)malloc(sizeof(size_t));
            ct->shape[0] = result.size();
            ct->data = malloc(result.size() * sizeof(int64_t));
            memcpy(ct->data, result.data(), result.size() * sizeof(int64_t));
            ct->elem_size = sizeof(int64_t);
            ct->dtype = IT_DTYPE_I64;
            ct->scale = 1.0f;
            ct->zero_point = 0.0f;
            ct->owns_memory = true;
            return ct;
        }
        // I8 直接用存储值比较
        case IT_DTYPE_I8: {
            auto t = to_cpp_tensor<I8>(input);
            auto result = argmax(t);
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
            ct->ndim = 1;
            ct->shape = (size_t*)malloc(sizeof(size_t));
            ct->shape[0] = result.size();
            ct->data = malloc(result.size() * sizeof(int64_t));
            memcpy(ct->data, result.data(), result.size() * sizeof(int64_t));
            ct->elem_size = sizeof(int64_t);
            ct->dtype = IT_DTYPE_I64;
            ct->scale = 1.0f;
            ct->zero_point = 0.0f;
            ct->owns_memory = true;
            return ct;
        }
        // F16, BF16 类似
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = argmax(t);
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
            ct->ndim = 1;
            ct->shape = (size_t*)malloc(sizeof(size_t));
            ct->shape[0] = result.size();
            ct->data = malloc(result.size() * sizeof(int64_t));
            memcpy(ct->data, result.data(), result.size() * sizeof(int64_t));
            ct->elem_size = sizeof(int64_t);
            ct->dtype = IT_DTYPE_I64;
            ct->scale = 1.0f;
            ct->zero_point = 0.0f;
            ct->owns_memory = true;
            return ct;
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = argmax(t);
            it_tensor* ct = (it_tensor*)malloc(sizeof(it_tensor));
            ct->ndim = 1;
            ct->shape = (size_t*)malloc(sizeof(size_t));
            ct->shape[0] = result.size();
            ct->data = malloc(result.size() * sizeof(int64_t));
            memcpy(ct->data, result.data(), result.size() * sizeof(int64_t));
            ct->elem_size = sizeof(int64_t);
            ct->dtype = IT_DTYPE_I64;
            ct->scale = 1.0f;
            ct->zero_point = 0.0f;
            ct->owns_memory = true;
            return ct;
        }
        default: return nullptr;
    }
}

// ============================================================
// topk
// ============================================================
// c_api_extra.cpp
extern "C" void it_topk(
    const it_tensor* input,
    size_t k,
    int dim,
    int largest,
    it_tensor** values,
    it_tensor** indices
) {
    if (!input || k == 0 || !values || !indices) {
        *values = nullptr;
        *indices = nullptr;
        return;
    }

    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = topk(t, k, dim, largest != 0, true);
            *values = from_cpp_tensor(result.first, IT_DTYPE_F32);
            // 索引转成 Tensor<I64>
            std::vector<size_t> out_shape = t.shape;
            out_shape[dim] = k;
            *indices = from_vector_to_tensor(result.second, out_shape, IT_DTYPE_I64);
            return;
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = topk(t, k, dim, largest != 0, true);
            *values = from_cpp_tensor(result.first, IT_DTYPE_F64);
            std::vector<size_t> out_shape = t.shape;
            out_shape[dim] = k;
            *indices = from_vector_to_tensor(result.second, out_shape, IT_DTYPE_I64);
            return;
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = topk(t, k, dim, largest != 0, true);
            *values = from_cpp_tensor(result.first, IT_DTYPE_F16);
            std::vector<size_t> out_shape = t.shape;
            out_shape[dim] = k;
            *indices = from_vector_to_tensor(result.second, out_shape, IT_DTYPE_I64);
            return;
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = topk(t, k, dim, largest != 0, true);
            *values = from_cpp_tensor(result.first, IT_DTYPE_BF16);
            std::vector<size_t> out_shape = t.shape;
            out_shape[dim] = k;
            *indices = from_vector_to_tensor(result.second, out_shape, IT_DTYPE_I64);
            return;
        }
        case IT_DTYPE_I8: {
            auto t = to_cpp_tensor<I8>(input);
            auto result = topk(t, k, dim, largest != 0, true);
            *values = from_cpp_tensor(result.first, IT_DTYPE_I8);
            std::vector<size_t> out_shape = t.shape;
            out_shape[dim] = k;
            *indices = from_vector_to_tensor(result.second, out_shape, IT_DTYPE_I64);
            return;
        }
        default:
            *values = nullptr;
            *indices = nullptr;
            return;
    }
}

// ============================================================
// gather
// ============================================================
extern "C" it_tensor* it_gather(
    const it_tensor* input,
    const int64_t* indices,
    size_t indices_len,
    const size_t* indices_shape,
    size_t indices_ndim,
    int dim
) {
    if (!input || !indices || indices_len == 0) return nullptr;

    // 将 indices 和 indices_shape 转换为 std::vector
    std::vector<int64_t> idx_vec(indices, indices + indices_len);
    std::vector<size_t> shape_vec(indices_shape, indices_shape + indices_ndim);

    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = gather(t, idx_vec, shape_vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = gather(t, idx_vec, shape_vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = gather(t, idx_vec, shape_vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = gather(t, idx_vec, shape_vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        case IT_DTYPE_I8: {
            auto t = to_cpp_tensor<I8>(input);
            auto result = gather(t, idx_vec, shape_vec, dim);
            return from_cpp_tensor(result, IT_DTYPE_I8);
        }
        default: return nullptr;
    }
}
