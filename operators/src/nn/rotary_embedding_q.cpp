#include "infer_train/nn/rotary_embedding_q.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<I8> quantized_rotary_embedding(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&);

} // namespace infer_train
