#include "infer_train/math/cat.hpp"
#include "infer_train/dtype.hpp"
#include "infer_train/tensor.hpp"

namespace infer_train {

template Tensor<F32> cat(const std::vector<const Tensor<F32>*>&, int);
template Tensor<F64> cat(const std::vector<const Tensor<F64>*>&, int);
template Tensor<F16> cat(const std::vector<const Tensor<F16>*>&, int);
template Tensor<BF16> cat(const std::vector<const Tensor<BF16>*>&, int);
template Tensor<I8> cat(const std::vector<const Tensor<I8>*>&, int);

} // namespace infer_train
