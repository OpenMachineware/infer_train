#include "infer_train/nn/conv_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_conv1d(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>*, int, int, int, int);
template Tensor<I8> quantized_conv3d(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>*, int, int, int, int);

} // namespace infer_train
