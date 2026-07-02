#include "c_api_common.h"
#include "infer_train/loss/cross_entropy.hpp"
#include "infer_train/loss/mse.hpp"
#include "infer_train/loss/l1.hpp"
#include "infer_train/loss/bce.hpp"
#include <vector>

using namespace infer_train;

extern "C" it_tensor* it_cross_entropy_loss(
    const it_tensor* input,
    const int64_t* target,
    size_t batch_size,
    int reduction
) {
    if (input->dtype != IT_DTYPE_F32 && input->dtype != IT_DTYPE_F64 &&
        input->dtype != IT_DTYPE_F16 && input->dtype != IT_DTYPE_BF16) {
        return nullptr;
    }

    std::vector<int64_t> target_vec(target, target + batch_size);

    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = cross_entropy_loss(t, target_vec, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = cross_entropy_loss(t, target_vec, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = cross_entropy_loss(t, target_vec, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = cross_entropy_loss(t, target_vec, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_mse_loss(
    const it_tensor* input,
    const it_tensor* target,
    int reduction
) {
    if (input->dtype != target->dtype) return nullptr;

    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto ti = to_cpp_tensor<F32>(input);
            auto tt = to_cpp_tensor<F32>(target);
            auto result = mse_loss(ti, tt, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto ti = to_cpp_tensor<F64>(input);
            auto tt = to_cpp_tensor<F64>(target);
            auto result = mse_loss(ti, tt, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto ti = to_cpp_tensor<F16>(input);
            auto tt = to_cpp_tensor<F16>(target);
            auto result = mse_loss(ti, tt, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto ti = to_cpp_tensor<BF16>(input);
            auto tt = to_cpp_tensor<BF16>(target);
            auto result = mse_loss(ti, tt, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_l1_loss(
    const it_tensor* input,
    const it_tensor* target,
    int reduction
) {
    if (input->dtype != target->dtype) return nullptr;

    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto ti = to_cpp_tensor<F32>(input);
            auto tt = to_cpp_tensor<F32>(target);
            auto result = l1_loss(ti, tt, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto ti = to_cpp_tensor<F64>(input);
            auto tt = to_cpp_tensor<F64>(target);
            auto result = l1_loss(ti, tt, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto ti = to_cpp_tensor<F16>(input);
            auto tt = to_cpp_tensor<F16>(target);
            auto result = l1_loss(ti, tt, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto ti = to_cpp_tensor<BF16>(input);
            auto tt = to_cpp_tensor<BF16>(target);
            auto result = l1_loss(ti, tt, reduction != 0);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

extern "C" it_tensor* it_bce_loss(
    const it_tensor* input,
    const it_tensor* target,
    int reduction,
    float eps
) {
    if (input->dtype != target->dtype) return nullptr;

    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto ti = to_cpp_tensor<F32>(input);
            auto tt = to_cpp_tensor<F32>(target);
            auto result = bce_loss(ti, tt, reduction != 0, eps);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto ti = to_cpp_tensor<F64>(input);
            auto tt = to_cpp_tensor<F64>(target);
            auto result = bce_loss(ti, tt, reduction != 0, eps);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto ti = to_cpp_tensor<F16>(input);
            auto tt = to_cpp_tensor<F16>(target);
            auto result = bce_loss(ti, tt, reduction != 0, eps);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto ti = to_cpp_tensor<BF16>(input);
            auto tt = to_cpp_tensor<BF16>(target);
            auto result = bce_loss(ti, tt, reduction != 0, eps);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}
