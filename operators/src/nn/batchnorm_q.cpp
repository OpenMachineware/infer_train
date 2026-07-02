#include "infer_train/nn/batchnorm_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_batchnorm2d_inference(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, float);
template Tensor<I8> quantized_batchnorm1d_inference(const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, const Tensor<I8>&, float);

}
