#include "infer_train/nn/rotary_embedding.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> rotary_embedding(const Tensor<F32>&, const Tensor<F32>&, const Tensor<F32>&);
template Tensor<F64> rotary_embedding(const Tensor<F64>&, const Tensor<F64>&, const Tensor<F64>&);
template Tensor<F16> rotary_embedding(const Tensor<F16>&, const Tensor<F16>&, const Tensor<F16>&);
template Tensor<BF16> rotary_embedding(const Tensor<BF16>&, const Tensor<BF16>&, const Tensor<BF16>&);

// ============================================================
// 输出参数版本（内部实现）
// ============================================================
template void precompute_rotary_embeddings_impl<F32>(size_t, size_t, float, Tensor<F32>&, Tensor<F32>&);
template void precompute_rotary_embeddings_impl<F64>(size_t, size_t, float, Tensor<F64>&, Tensor<F64>&);
template void precompute_rotary_embeddings_impl<F16>(size_t, size_t, float, Tensor<F16>&, Tensor<F16>&);
template void precompute_rotary_embeddings_impl<BF16>(size_t, size_t, float, Tensor<BF16>&, Tensor<BF16>&);

// ============================================================
// pair 版本（对外使用）
// ============================================================
template std::pair<Tensor<F32>, Tensor<F32>> precompute_rotary_embeddings<F32>(size_t, size_t, float);
template std::pair<Tensor<F64>, Tensor<F64>> precompute_rotary_embeddings<F64>(size_t, size_t, float);
template std::pair<Tensor<F16>, Tensor<F16>> precompute_rotary_embeddings<F16>(size_t, size_t, float);
template std::pair<Tensor<BF16>, Tensor<BF16>> precompute_rotary_embeddings<BF16>(size_t, size_t, float);

} // namespace infer_train
