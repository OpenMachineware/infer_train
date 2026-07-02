#include "infer_train/nn/relu_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_relu(const Tensor<I8>&);
template Tensor<I8> quantized_leaky_relu(const Tensor<I8>&, float);
template Tensor<I8> quantized_elu(const Tensor<I8>&, float);
template Tensor<I8> quantized_gelu(const Tensor<I8>&);
}
