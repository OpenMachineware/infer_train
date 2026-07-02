#include "infer_train/c_interface.h"
#include "infer_train/nn.hpp"
#include "c_api_common.h"

using namespace infer_train;

// ============================================================
// 分派宏：浮点 NN 算子（带多个参数）
// ============================================================
#define DISPATCH_NN_CONV2D \
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

// c_api_nn.cpp 中已有 DISPATCH_NN_CONV2D，复制一份改参数数量
#define DISPATCH_NN_CONV1D \
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

#define DISPATCH_NN_CONV3D \
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
// conv2d
// ============================================================
extern "C" it_tensor* it_conv2d(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    DISPATCH_NN_CONV2D;
}

// ============================================================
// conv1d
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
    DISPATCH_NN_CONV1D;  // 类似 conv2d 的分派
}

// 同样添加 conv3d, maxpool1d, maxpool3d, avgpool1d, avgpool3d,
// batchnorm1d_inference, instancenorm2d, groupnorm, dropout, embedding

// ============================================================
// maxpool2d
// ============================================================
extern "C" it_tensor* it_maxpool2d(
    const it_tensor* input,
    int kernel_size,
    int stride,
    int padding
) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = maxpool2d(t, kernel_size, stride, padding);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = maxpool2d(t, kernel_size, stride, padding);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = maxpool2d(t, kernel_size, stride, padding);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = maxpool2d(t, kernel_size, stride, padding);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// avgpool2d
// ============================================================
extern "C" it_tensor* it_avgpool2d(
    const it_tensor* input,
    int kernel_size,
    int stride,
    int padding
) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto t = to_cpp_tensor<F32>(input);
            auto result = avgpool2d(t, kernel_size, stride, padding);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto t = to_cpp_tensor<F64>(input);
            auto result = avgpool2d(t, kernel_size, stride, padding);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto t = to_cpp_tensor<F16>(input);
            auto result = avgpool2d(t, kernel_size, stride, padding);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto t = to_cpp_tensor<BF16>(input);
            auto result = avgpool2d(t, kernel_size, stride, padding);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// batchnorm2d_inference
// ============================================================
extern "C" it_tensor* it_batchnorm2d_inference(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    const it_tensor* running_mean,
    const it_tensor* running_var,
    float eps
) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto ti = to_cpp_tensor<F32>(input);
            auto tw = to_cpp_tensor<F32>(weight);
            auto tb = to_cpp_tensor<F32>(bias);
            auto tm = to_cpp_tensor<F32>(running_mean);
            auto tv = to_cpp_tensor<F32>(running_var);
            auto result = batchnorm2d_inference(ti, tw, tb, tm, tv, eps);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto ti = to_cpp_tensor<F64>(input);
            auto tw = to_cpp_tensor<F64>(weight);
            auto tb = to_cpp_tensor<F64>(bias);
            auto tm = to_cpp_tensor<F64>(running_mean);
            auto tv = to_cpp_tensor<F64>(running_var);
            auto result = batchnorm2d_inference(ti, tw, tb, tm, tv, eps);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto ti = to_cpp_tensor<F16>(input);
            auto tw = to_cpp_tensor<F16>(weight);
            auto tb = to_cpp_tensor<F16>(bias);
            auto tm = to_cpp_tensor<F16>(running_mean);
            auto tv = to_cpp_tensor<F16>(running_var);
            auto result = batchnorm2d_inference(ti, tw, tb, tm, tv, eps);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto ti = to_cpp_tensor<BF16>(input);
            auto tw = to_cpp_tensor<BF16>(weight);
            auto tb = to_cpp_tensor<BF16>(bias);
            auto tm = to_cpp_tensor<BF16>(running_mean);
            auto tv = to_cpp_tensor<BF16>(running_var);
            auto result = batchnorm2d_inference(ti, tw, tb, tm, tv, eps);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// layernorm
// ============================================================
extern "C" it_tensor* it_layernorm(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias,
    float eps
) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto ti = to_cpp_tensor<F32>(input);
            auto tw = to_cpp_tensor<F32>(weight);
            auto tb = to_cpp_tensor<F32>(bias);
            auto result = layernorm(ti, tw, tb, eps);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto ti = to_cpp_tensor<F64>(input);
            auto tw = to_cpp_tensor<F64>(weight);
            auto tb = to_cpp_tensor<F64>(bias);
            auto result = layernorm(ti, tw, tb, eps);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto ti = to_cpp_tensor<F16>(input);
            auto tw = to_cpp_tensor<F16>(weight);
            auto tb = to_cpp_tensor<F16>(bias);
            auto result = layernorm(ti, tw, tb, eps);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto ti = to_cpp_tensor<BF16>(input);
            auto tw = to_cpp_tensor<BF16>(weight);
            auto tb = to_cpp_tensor<BF16>(bias);
            auto result = layernorm(ti, tw, tb, eps);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}

// ============================================================
// linear
// ============================================================
extern "C" it_tensor* it_linear(
    const it_tensor* input,
    const it_tensor* weight,
    const it_tensor* bias
) {
    switch (input->dtype) {
        case IT_DTYPE_F32: {
            auto ti = to_cpp_tensor<F32>(input);
            auto tw = to_cpp_tensor<F32>(weight);
            Tensor<F32> tb; bool has_bias = (bias != nullptr);
            if (has_bias) tb = to_cpp_tensor<F32>(bias);
            auto result = linear(ti, tw, has_bias ? &tb : nullptr);
            return from_cpp_tensor(result, IT_DTYPE_F32);
        }
        case IT_DTYPE_F64: {
            auto ti = to_cpp_tensor<F64>(input);
            auto tw = to_cpp_tensor<F64>(weight);
            Tensor<F64> tb; bool has_bias = (bias != nullptr);
            if (has_bias) tb = to_cpp_tensor<F64>(bias);
            auto result = linear(ti, tw, has_bias ? &tb : nullptr);
            return from_cpp_tensor(result, IT_DTYPE_F64);
        }
        case IT_DTYPE_F16: {
            auto ti = to_cpp_tensor<F16>(input);
            auto tw = to_cpp_tensor<F16>(weight);
            Tensor<F16> tb; bool has_bias = (bias != nullptr);
            if (has_bias) tb = to_cpp_tensor<F16>(bias);
            auto result = linear(ti, tw, has_bias ? &tb : nullptr);
            return from_cpp_tensor(result, IT_DTYPE_F16);
        }
        case IT_DTYPE_BF16: {
            auto ti = to_cpp_tensor<BF16>(input);
            auto tw = to_cpp_tensor<BF16>(weight);
            Tensor<BF16> tb; bool has_bias = (bias != nullptr);
            if (has_bias) tb = to_cpp_tensor<BF16>(bias);
            auto result = linear(ti, tw, has_bias ? &tb : nullptr);
            return from_cpp_tensor(result, IT_DTYPE_BF16);
        }
        default: return nullptr;
    }
}
