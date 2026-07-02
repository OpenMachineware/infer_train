#include "infer_train/math/matmul_q.hpp"

namespace infer_train {
template Tensor<I8> quantized_matmul(const Tensor<I8>&, const Tensor<I8>&);
template Tensor<I8> quantized_vec_matmul(const Tensor<I8>&, const Tensor<I8>&);
template Tensor<I8> quantized_transpose(const Tensor<I8>&);
}
