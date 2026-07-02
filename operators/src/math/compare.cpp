#include "infer_train/math/compare.hpp"

namespace infer_train {

template Tensor<uint8_t> eq(const Tensor<F32>&, const Tensor<F32>&);
template Tensor<uint8_t> eq(const Tensor<F64>&, const Tensor<F64>&);
template Tensor<uint8_t> eq(const Tensor<F16>&, const Tensor<F16>&);
template Tensor<uint8_t> eq(const Tensor<BF16>&, const Tensor<BF16>&);

template Tensor<uint8_t> ne(const Tensor<F32>&, const Tensor<F32>&);
template Tensor<uint8_t> ne(const Tensor<F64>&, const Tensor<F64>&);
template Tensor<uint8_t> ne(const Tensor<F16>&, const Tensor<F16>&);
template Tensor<uint8_t> ne(const Tensor<BF16>&, const Tensor<BF16>&);

template Tensor<uint8_t> gt(const Tensor<F32>&, const Tensor<F32>&);
template Tensor<uint8_t> gt(const Tensor<F64>&, const Tensor<F64>&);
template Tensor<uint8_t> gt(const Tensor<F16>&, const Tensor<F16>&);
template Tensor<uint8_t> gt(const Tensor<BF16>&, const Tensor<BF16>&);

template Tensor<uint8_t> lt(const Tensor<F32>&, const Tensor<F32>&);
template Tensor<uint8_t> lt(const Tensor<F64>&, const Tensor<F64>&);
template Tensor<uint8_t> lt(const Tensor<F16>&, const Tensor<F16>&);
template Tensor<uint8_t> lt(const Tensor<BF16>&, const Tensor<BF16>&);

template Tensor<uint8_t> ge(const Tensor<F32>&, const Tensor<F32>&);
template Tensor<uint8_t> ge(const Tensor<F64>&, const Tensor<F64>&);
template Tensor<uint8_t> ge(const Tensor<F16>&, const Tensor<F16>&);
template Tensor<uint8_t> ge(const Tensor<BF16>&, const Tensor<BF16>&);

template Tensor<uint8_t> le(const Tensor<F32>&, const Tensor<F32>&);
template Tensor<uint8_t> le(const Tensor<F64>&, const Tensor<F64>&);
template Tensor<uint8_t> le(const Tensor<F16>&, const Tensor<F16>&);
template Tensor<uint8_t> le(const Tensor<BF16>&, const Tensor<BF16>&);

} // namespace infer_train
