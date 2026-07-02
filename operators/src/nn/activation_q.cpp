#include "infer_train/nn/activation_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_sigmoid(const Tensor<I8>&);
template Tensor<I8> quantized_tanh(const Tensor<I8>&);
template Tensor<I8> quantized_silu(const Tensor<I8>&);
template Tensor<I8> quantized_hard_swish(const Tensor<I8>&);
template Tensor<I8> quantized_hard_sigmoid(const Tensor<I8>&);

template Tensor<I8> quantized_softplus(const Tensor<I8>&, float, float);
template Tensor<I8> quantized_softshrink(const Tensor<I8>&, float);
template Tensor<I8> quantized_celu(const Tensor<I8>&, float);

template Tensor<I8> quantized_relu6(const Tensor<I8>&);


}
