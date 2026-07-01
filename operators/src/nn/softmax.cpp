#include "infer_train/nn/softmax.hpp"

namespace infer_train {

    // FP32
    template Tensor<F32> softmax(const Tensor<F32>&, int);
    template Tensor<F32> log_softmax(const Tensor<F32>&, int);

    // FP64
    template Tensor<F64> softmax(const Tensor<F64>&, int);
    template Tensor<F64> log_softmax(const Tensor<F64>&, int);

    // FP16
    template Tensor<F16> softmax(const Tensor<F16>&, int);
    template Tensor<F16> log_softmax(const Tensor<F16>&, int);

    // BF16
    template Tensor<BF16> softmax(const Tensor<BF16>&, int);
    template Tensor<BF16> log_softmax(const Tensor<BF16>&, int);

} // namespace infer_train
