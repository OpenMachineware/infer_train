#include "infer_train/math/std_var_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_var(const Tensor<I8>&, bool);
template Tensor<I8> quantized_std(const Tensor<I8>&, bool);

} // namespace infer_train
