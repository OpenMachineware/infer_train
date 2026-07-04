#include "c_api_common.h"
#include "infer_train/nn.hpp"

using namespace infer_train;

// ============================================================
// conv1d
// ============================================================
#define DISPATCH_CONV1D \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            Tensor<F32> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F32>(bias); \
            auto result = conv1d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            Tensor<F64> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F64>(bias); \
            auto result = conv1d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            Tensor<F16> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F16>(bias); \
            auto result = conv1d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            Tensor<BF16> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<BF16>(bias); \
            auto result = conv1d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

// ============================================================
// conv2d
// ============================================================
#define DISPATCH_CONV2D \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            Tensor<F32> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F32>(bias); \
            auto result = conv2d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            Tensor<F64> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F64>(bias); \
            auto result = conv2d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            Tensor<F16> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F16>(bias); \
            auto result = conv2d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            Tensor<BF16> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<BF16>(bias); \
            auto result = conv2d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

// ============================================================
// conv3d
// ============================================================
#define DISPATCH_CONV3D \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            Tensor<F32> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F32>(bias); \
            auto result = conv3d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            Tensor<F64> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F64>(bias); \
            auto result = conv3d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            Tensor<F16> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F16>(bias); \
            auto result = conv3d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            Tensor<BF16> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<BF16>(bias); \
            auto result = conv3d(ti, tw, has_bias ? &tb : nullptr, stride, padding, dilation, groups); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

// ============================================================
// 池化分派
// ============================================================
#define DISPATCH_POOL_1D(op) \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto t = to_cpp_tensor<F32>(input); \
            auto result = op(t, kernel_size, stride, padding); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto t = to_cpp_tensor<F64>(input); \
            auto result = op(t, kernel_size, stride, padding); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto t = to_cpp_tensor<F16>(input); \
            auto result = op(t, kernel_size, stride, padding); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto t = to_cpp_tensor<BF16>(input); \
            auto result = op(t, kernel_size, stride, padding); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

#define DISPATCH_POOL_2D(op) DISPATCH_POOL_1D(op)
#define DISPATCH_POOL_3D(op) DISPATCH_POOL_1D(op)

// ============================================================
// NN 算子实现
// ============================================================
extern "C" it_tensor* it_conv1d(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    DISPATCH_CONV1D;
}

extern "C" it_tensor* it_conv2d(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    DISPATCH_CONV2D;
}

extern "C" it_tensor* it_conv3d(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    DISPATCH_CONV3D;
}

// ============================================================
// 池化
// ============================================================
extern "C" it_tensor* it_maxpool1d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_POOL_1D(maxpool1d);
}

extern "C" it_tensor* it_maxpool2d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_POOL_2D(maxpool2d);
}

extern "C" it_tensor* it_maxpool3d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_POOL_3D(maxpool3d);
}

extern "C" it_tensor* it_avgpool1d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_POOL_1D(avgpool1d);
}

extern "C" it_tensor* it_avgpool2d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_POOL_2D(avgpool2d);
}

extern "C" it_tensor* it_avgpool3d(const it_tensor* input, int kernel_size, int stride, int padding) {
    DISPATCH_POOL_3D(avgpool3d);
}

// ============================================================
// 归一化
// ============================================================
#define DISPATCH_BATCHNORM1D \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            auto tb = to_cpp_tensor<F32>(bias); \
            auto tm = to_cpp_tensor<F32>(running_mean); \
            auto tv = to_cpp_tensor<F32>(running_var); \
            auto result = batchnorm1d_inference(ti, tw, tb, tm, tv, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            auto tb = to_cpp_tensor<F64>(bias); \
            auto tm = to_cpp_tensor<F64>(running_mean); \
            auto tv = to_cpp_tensor<F64>(running_var); \
            auto result = batchnorm1d_inference(ti, tw, tb, tm, tv, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            auto tb = to_cpp_tensor<F16>(bias); \
            auto tm = to_cpp_tensor<F16>(running_mean); \
            auto tv = to_cpp_tensor<F16>(running_var); \
            auto result = batchnorm1d_inference(ti, tw, tb, tm, tv, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            auto tb = to_cpp_tensor<BF16>(bias); \
            auto tm = to_cpp_tensor<BF16>(running_mean); \
            auto tv = to_cpp_tensor<BF16>(running_var); \
            auto result = batchnorm1d_inference(ti, tw, tb, tm, tv, eps); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

#define DISPATCH_BATCHNORM2D \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            auto tb = to_cpp_tensor<F32>(bias); \
            auto tm = to_cpp_tensor<F32>(running_mean); \
            auto tv = to_cpp_tensor<F32>(running_var); \
            auto result = batchnorm2d_inference(ti, tw, tb, tm, tv, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            auto tb = to_cpp_tensor<F64>(bias); \
            auto tm = to_cpp_tensor<F64>(running_mean); \
            auto tv = to_cpp_tensor<F64>(running_var); \
            auto result = batchnorm2d_inference(ti, tw, tb, tm, tv, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            auto tb = to_cpp_tensor<F16>(bias); \
            auto tm = to_cpp_tensor<F16>(running_mean); \
            auto tv = to_cpp_tensor<F16>(running_var); \
            auto result = batchnorm2d_inference(ti, tw, tb, tm, tv, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            auto tb = to_cpp_tensor<BF16>(bias); \
            auto tm = to_cpp_tensor<BF16>(running_mean); \
            auto tv = to_cpp_tensor<BF16>(running_var); \
            auto result = batchnorm2d_inference(ti, tw, tb, tm, tv, eps); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

#define DISPATCH_NORM_2D(op) \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            auto tb = to_cpp_tensor<F32>(bias); \
            auto result = op(ti, tw, tb, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            auto tb = to_cpp_tensor<F64>(bias); \
            auto result = op(ti, tw, tb, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            auto tb = to_cpp_tensor<F16>(bias); \
            auto result = op(ti, tw, tb, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            auto tb = to_cpp_tensor<BF16>(bias); \
            auto result = op(ti, tw, tb, eps); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

#define DISPATCH_RMSNORM \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            auto result = rmsnorm(ti, tw, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            auto result = rmsnorm(ti, tw, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            auto result = rmsnorm(ti, tw, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            auto result = rmsnorm(ti, tw, eps); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

#define DISPATCH_GROUPNORM \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            auto tb = to_cpp_tensor<F32>(bias); \
            auto result = groupnorm(ti, tw, tb, num_groups, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            auto tb = to_cpp_tensor<F64>(bias); \
            auto result = groupnorm(ti, tw, tb, num_groups, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            auto tb = to_cpp_tensor<F16>(bias); \
            auto result = groupnorm(ti, tw, tb, num_groups, eps); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            auto tb = to_cpp_tensor<BF16>(bias); \
            auto result = groupnorm(ti, tw, tb, num_groups, eps); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

extern "C" it_tensor* it_batchnorm1d_inference(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    const it_tensor* running_mean,
    const it_tensor* running_var,
    float eps
) {
    DISPATCH_BATCHNORM1D;
}

extern "C" it_tensor* it_batchnorm2d_inference(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    const it_tensor* running_mean,
    const it_tensor* running_var,
    float eps
) {
    DISPATCH_BATCHNORM2D;
}

extern "C" it_tensor* it_layernorm(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    float eps
) {
    DISPATCH_NORM_2D(layernorm);
}

extern "C" it_tensor* it_rmsnorm(
    const it_tensor* input,
    const it_tensor* weight,
    float eps
) {
    DISPATCH_RMSNORM;
}

extern "C" it_tensor* it_instancenorm2d(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    float eps
) {
    DISPATCH_NORM_2D(instancenorm2d);
}

extern "C" it_tensor* it_groupnorm(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    int num_groups,
    float eps
) {
    DISPATCH_GROUPNORM;
}

// ============================================================
// linear
// ============================================================
#define DISPATCH_LINEAR \
    switch (input->dtype) { \
        case IT_DTYPE_F32: { \
            auto ti = to_cpp_tensor<F32>(input); \
            auto tw = to_cpp_tensor<F32>(weight); \
            Tensor<F32> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F32>(bias); \
            auto result = linear(ti, tw, has_bias ? &tb : nullptr); \
            return from_cpp_tensor(result, IT_DTYPE_F32); \
        } \
        case IT_DTYPE_F64: { \
            auto ti = to_cpp_tensor<F64>(input); \
            auto tw = to_cpp_tensor<F64>(weight); \
            Tensor<F64> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F64>(bias); \
            auto result = linear(ti, tw, has_bias ? &tb : nullptr); \
            return from_cpp_tensor(result, IT_DTYPE_F64); \
        } \
        case IT_DTYPE_F16: { \
            auto ti = to_cpp_tensor<F16>(input); \
            auto tw = to_cpp_tensor<F16>(weight); \
            Tensor<F16> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<F16>(bias); \
            auto result = linear(ti, tw, has_bias ? &tb : nullptr); \
            return from_cpp_tensor(result, IT_DTYPE_F16); \
        } \
        case IT_DTYPE_BF16: { \
            auto ti = to_cpp_tensor<BF16>(input); \
            auto tw = to_cpp_tensor<BF16>(weight); \
            Tensor<BF16> tb; bool has_bias = (bias != nullptr); \
            if (has_bias) tb = to_cpp_tensor<BF16>(bias); \
            auto result = linear(ti, tw, has_bias ? &tb : nullptr); \
            return from_cpp_tensor(result, IT_DTYPE_BF16); \
        } \
        default: return nullptr; \
    }

extern "C" it_tensor* it_linear(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias
) {
    DISPATCH_LINEAR;
}

// ============================================================
// embedding
// ============================================================
extern "C" it_tensor* it_embedding(
    const int64_t* indices,
    size_t num_indices,
    const it_tensor* weight,
    int padding_idx
) {
    if (weight->dtype != IT_DTYPE_F32 && weight->dtype != IT_DTYPE_F64 &&
        weight->dtype != IT_DTYPE_F16 && weight->dtype != IT_DTYPE_BF16) {
        return nullptr;
        }

    std::vector<int64_t> idx_vec(indices, indices + num_indices);

    switch (weight->dtype) {
        case IT_DTYPE_F32: {
            auto tw = to_cpp_tensor<F32>(weight);
            auto result = embedding(idx_vec, tw, padding_idx);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto tw = to_cpp_tensor<F64>(weight);
            auto result = embedding(idx_vec, tw, padding_idx);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto tw = to_cpp_tensor<F16>(weight);
            auto result = embedding(idx_vec, tw, padding_idx);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto tw = to_cpp_tensor<BF16>(weight);
            auto result = embedding(idx_vec, tw, padding_idx);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// dropout（推理模式）
// ============================================================
extern "C" it_tensor* it_dropout(
    const it_tensor* input,
    float p,
    int training,
    uint32_t seed
) {
    if (training) {
        switch (input->dtype) {
            case IT_DTYPE_F32: {
                auto t = to_cpp_tensor<F32>(input);
                auto result = dropout_training(t, p, seed);
                return from_cpp_tensor(result, IT_DTYPE_F32);
            }
            case IT_DTYPE_F64: {
                auto t = to_cpp_tensor<F64>(input);
                auto result = dropout_training(t, p, seed);
                return from_cpp_tensor(result, IT_DTYPE_F64);
            }
            case IT_DTYPE_F16: {
                auto t = to_cpp_tensor<F16>(input);
                auto result = dropout_training(t, p, seed);
                return from_cpp_tensor(result, IT_DTYPE_F16);
            }
            case IT_DTYPE_BF16: {
                auto t = to_cpp_tensor<BF16>(input);
                auto result = dropout_training(t, p, seed);
                return from_cpp_tensor(result, IT_DTYPE_BF16);
            }
            default: return nullptr;
        }
    } else {
        switch (input->dtype) {
            case IT_DTYPE_F32: {
                auto t = to_cpp_tensor<F32>(input);
                auto result = dropout_inference(t, p);
                return from_cpp_tensor(result, IT_DTYPE_F32);
            }
            case IT_DTYPE_F64: {
                auto t = to_cpp_tensor<F64>(input);
                auto result = dropout_inference(t, p);
                return from_cpp_tensor(result, IT_DTYPE_F64);
            }
            case IT_DTYPE_F16: {
                auto t = to_cpp_tensor<F16>(input);
                auto result = dropout_inference(t, p);
                return from_cpp_tensor(result, IT_DTYPE_F16);
            }
            case IT_DTYPE_BF16: {
                auto t = to_cpp_tensor<BF16>(input);
                auto result = dropout_inference(t, p);
                return from_cpp_tensor(result, IT_DTYPE_BF16);
            }
            default: return nullptr;
        }
    }
}
