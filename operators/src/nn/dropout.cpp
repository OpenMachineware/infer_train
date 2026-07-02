#include "infer_train/nn/dropout.hpp"

namespace infer_train {

template Tensor<F32> dropout_inference(const Tensor<F32>&, float);
template Tensor<F64> dropout_inference(const Tensor<F64>&, float);
template Tensor<F16> dropout_inference(const Tensor<F16>&, float);
template Tensor<BF16> dropout_inference(const Tensor<BF16>&, float);

} // namespace infer_train
