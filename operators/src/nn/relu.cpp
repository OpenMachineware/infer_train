#include "infer_train/nn/relu.hpp"

namespace infer_train {

    // FP32
    template Tensor<F32> relu(const Tensor<F32>&);
    template Tensor<F32> leaky_relu(const Tensor<F32>&, float);
    template Tensor<F32> elu(const Tensor<F32>&, float);
    template Tensor<F32> gelu(const Tensor<F32>&);

    // FP64
    template Tensor<F64> relu(const Tensor<F64>&);
    template Tensor<F64> leaky_relu(const Tensor<F64>&, float);
    template Tensor<F64> elu(const Tensor<F64>&, float);
    template Tensor<F64> gelu(const Tensor<F64>&);

    // FP16
    template Tensor<F16> relu(const Tensor<F16>&);
    template Tensor<F16> leaky_relu(const Tensor<F16>&, float);
    template Tensor<F16> elu(const Tensor<F16>&, float);
    template Tensor<F16> gelu(const Tensor<F16>&);

    // BF16
    template Tensor<BF16> relu(const Tensor<BF16>&);
    template Tensor<BF16> leaky_relu(const Tensor<BF16>&, float);
    template Tensor<BF16> elu(const Tensor<BF16>&, float);
    template Tensor<BF16> gelu(const Tensor<BF16>&);

} // namespace infer_train