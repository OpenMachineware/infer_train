#include "c_api_common.h"
#include "infer_train/math.hpp"
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
