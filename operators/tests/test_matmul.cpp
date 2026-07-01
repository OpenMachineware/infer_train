#include <iostream>
#include "infer_train/math/matmul.hpp"

using namespace infer_train;

int main() {
    // 测试 2x3 * 3x2 = 2x2
    Tensor<float> a({2, 3});
    a.data = {1, 2, 3, 4, 5, 6};

    Tensor<float> b({3, 2});
    b.data = {7, 8, 9, 10, 11, 12};

    Tensor<float> c = matmul(a, b);

    std::cout << "Result:" << std::endl;
    for (size_t i = 0; i < c.shape[0]; ++i) {
        for (size_t j = 0; j < c.shape[1]; ++j) {
            std::cout << c.data[i * c.shape[1] + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
