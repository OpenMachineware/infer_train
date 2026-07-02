#include "infer_train/nn/activation.hpp"

namespace infer_train {

// ============================================================
// FP32
// ============================================================
template Tensor<F32> sigmoid(const Tensor<F32>&);
template Tensor<F32> tanh(const Tensor<F32>&);
template Tensor<F32> hard_sigmoid(const Tensor<F32>&);
template Tensor<F32> hard_swish(const Tensor<F32>&);
template Tensor<F32> silu(const Tensor<F32>&);

// ============================================================
// FP64
// ============================================================
template Tensor<F64> sigmoid(const Tensor<F64>&);
template Tensor<F64> tanh(const Tensor<F64>&);
template Tensor<F64> hard_sigmoid(const Tensor<F64>&);
template Tensor<F64> hard_swish(const Tensor<F64>&);
template Tensor<F64> silu(const Tensor<F64>&);

// ============================================================
// FP16
// ============================================================
template Tensor<F16> sigmoid(const Tensor<F16>&);
template Tensor<F16> tanh(const Tensor<F16>&);
template Tensor<F16> hard_sigmoid(const Tensor<F16>&);
template Tensor<F16> hard_swish(const Tensor<F16>&);
template Tensor<F16> silu(const Tensor<F16>&);

// ============================================================
// BF16
// ============================================================
template Tensor<BF16> sigmoid(const Tensor<BF16>&);
template Tensor<BF16> tanh(const Tensor<BF16>&);
template Tensor<BF16> hard_sigmoid(const Tensor<BF16>&);
template Tensor<BF16> hard_swish(const Tensor<BF16>&);
template Tensor<BF16> silu(const Tensor<BF16>&);

// softplus
template Tensor<F32> softplus(const Tensor<F32>&, float, float);
template Tensor<F64> softplus(const Tensor<F64>&, float, float);
template Tensor<F16> softplus(const Tensor<F16>&, float, float);
template Tensor<BF16> softplus(const Tensor<BF16>&, float, float);

// softshrink
template Tensor<F32> softshrink(const Tensor<F32>&, float);
template Tensor<F64> softshrink(const Tensor<F64>&, float);
template Tensor<F16> softshrink(const Tensor<F16>&, float);
template Tensor<BF16> softshrink(const Tensor<BF16>&, float);

// celu
template Tensor<F32> celu(const Tensor<F32>&, float);
template Tensor<F64> celu(const Tensor<F64>&, float);
template Tensor<F16> celu(const Tensor<F16>&, float);
template Tensor<BF16> celu(const Tensor<BF16>&, float);

template Tensor<F32> relu6(const Tensor<F32>&);
template Tensor<F64> relu6(const Tensor<F64>&);
template Tensor<F16> relu6(const Tensor<F16>&);
template Tensor<BF16> relu6(const Tensor<BF16>&);


} // namespace infer_train
