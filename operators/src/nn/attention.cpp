#include "infer_train/nn/attention.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> scaled_dot_product_attention(
    const Tensor<F32>&,
    const Tensor<F32>&,
    const Tensor<F32>&,
    const Tensor<F32>*,
    float,
    bool,
    float
);

template Tensor<F64> scaled_dot_product_attention(
    const Tensor<F64>&,
    const Tensor<F64>&,
    const Tensor<F64>&,
    const Tensor<F64>*,
    float,
    bool,
    float
);

template Tensor<F16> scaled_dot_product_attention(
    const Tensor<F16>&,
    const Tensor<F16>&,
    const Tensor<F16>&,
    const Tensor<F16>*,
    float,
    bool,
    float
);

template Tensor<BF16> scaled_dot_product_attention(
    const Tensor<BF16>&,
    const Tensor<BF16>&,
    const Tensor<BF16>&,
    const Tensor<BF16>*,
    float,
    bool,
    float
);

template Tensor<F32> multi_head_attention(
    const Tensor<F32>&,
    const Tensor<F32>&,
    const Tensor<F32>&,
    const Tensor<F32>*,
    int,
    float,
    bool,
    float
);

template Tensor<F64> multi_head_attention(
    const Tensor<F64>&,
    const Tensor<F64>&,
    const Tensor<F64>&,
    const Tensor<F64>*,
    int,
    float,
    bool,
    float
);

template Tensor<F16> multi_head_attention(
    const Tensor<F16>&,
    const Tensor<F16>&,
    const Tensor<F16>&,
    const Tensor<F16>*,
    int,
    float,
    bool,
    float
);

template Tensor<BF16> multi_head_attention(
    const Tensor<BF16>&,
    const Tensor<BF16>&,
    const Tensor<BF16>&,
    const Tensor<BF16>*,
    int,
    float,
    bool,
    float
);

} // namespace infer_train
