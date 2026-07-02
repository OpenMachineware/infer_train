#pragma once
#include "infer_train/tensor.hpp"
#include "infer_train/dtype.hpp"
#include <cmath>

namespace infer_train {

// ============================================================
// Sigmoid
// ============================================================
template<typename T>
Tensor<T> sigmoid(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(1.0f / (1.0f + std::exp(-x)));
    }
    
    return result;
}

// ============================================================
// Tanh
// ============================================================
template<typename T>
Tensor<T> tanh(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::tanh(x));
    }
    
    return result;
}

// ============================================================
// Hard Sigmoid
// ============================================================
template<typename T>
Tensor<T> hard_sigmoid(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float y = std::clamp(0.2f * x + 0.5f, 0.0f, 1.0f);
        result.data[i] = Conv::from_float(y);
    }
    
    return result;
}

// ============================================================
// Hard Swish
// ============================================================
template<typename T>
Tensor<T> hard_swish(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float y = x * std::clamp(0.2f * x + 0.5f, 0.0f, 1.0f);
        result.data[i] = Conv::from_float(y);
    }
    
    return result;
}

// ============================================================
// SiLU (Swish)
// ============================================================
template<typename T>
Tensor<T> silu(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(x * (1.0f / (1.0f + std::exp(-x))));
    }
    
    return result;
}

// ============================================================
// Softplus: log(1 + exp(x))
// ============================================================
template<typename T>
Tensor<T> softplus(const Tensor<T>& input, float beta = 1.0f, float threshold = 20.0f) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float val;
        if (x * beta > threshold) {
            val = x;
        } else {
            val = std::log(1.0f + std::exp(x * beta)) / beta;
        }
        result.data[i] = Conv::from_float(val);
    }
    return result;
}

// ============================================================
// Softshrink: x - lambda if x > lambda, x + lambda if x < -lambda, 0 otherwise
// ============================================================
template<typename T>
Tensor<T> softshrink(const Tensor<T>& input, float lambda = 0.5f) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float val;
        if (x > lambda) {
            val = x - lambda;
        } else if (x < -lambda) {
            val = x + lambda;
        } else {
            val = 0.0f;
        }
        result.data[i] = Conv::from_float(val);
    }
    return result;
}

// ============================================================
// CELU: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
// ============================================================
template<typename T>
Tensor<T> celu(const Tensor<T>& input, float alpha = 1.0f) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        float val = (x > 0.0f) ? x : alpha * (std::exp(x / alpha) - 1.0f);
        result.data[i] = Conv::from_float(val);
    }
    return result;
}

// ============================================================
// ReLU6: min(max(0, x), 6)
// ============================================================
template<typename T>
Tensor<T> relu6(const Tensor<T>& input) {
    using Conv = DTypeConverter<T>;
    Tensor<T> result(input.shape);
    for (size_t i = 0; i < result.size(); ++i) {
        float x = Conv::to_float(input.data[i]);
        result.data[i] = Conv::from_float(std::clamp(x, 0.0f, 6.0f));
    }
    return result;
}

} // namespace infer_train
