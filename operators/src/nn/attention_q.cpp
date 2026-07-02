#include "infer_train/nn/attention_q.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<I8> quantized_scaled_dot_product_attention(
    const Tensor<I8>&,
    const Tensor<I8>&,
    const Tensor<I8>&,
    const Tensor<I8>*,
    float,
    bool,
    float
);

template Tensor<I8> quantized_multi_head_attention(
    const Tensor<I8>&,
    const Tensor<I8>&,
    const Tensor<I8>&,
    const Tensor<I8>*,
    int,
    float,
    bool,
    float
);

} // namespace infer_train
