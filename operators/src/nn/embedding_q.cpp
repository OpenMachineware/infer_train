#include "infer_train/nn/embedding_q.hpp"

namespace infer_train {

template Tensor<I8> quantized_embedding(const std::vector<int64_t>&, const Tensor<I8>&, int);

}
