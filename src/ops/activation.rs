// src/ops/activation/mod.rs

pub mod elu;
pub mod gelu;
pub mod hard_swish;
pub mod leaky_relu;
pub mod log_softmax;
pub mod relu;
pub mod relu6;
pub mod sigmoid;
pub mod silu;
pub mod softmax;
pub mod tanh;

pub use elu::{
    elu, elu_backward, quantized_elu, quantized_elu_backward, EluOp,
    QuantizedEluOp,
};
pub use gelu::{
    gelu, gelu_backward, quantized_gelu, quantized_gelu_backward, GeluOp,
    QuantizedGeluOp,
};
pub use hard_swish::{
    hard_swish, hard_swish_backward, quantized_hard_swish,
    quantized_hard_swish_backward, HardSwishOp, QuantizedHardSwishOp,
};
pub use leaky_relu::{
    leaky_relu, leaky_relu_backward, quantized_leaky_relu,
    quantized_leaky_relu_backward, LeakyReluOp, QuantizedLeakyReluOp,
};
pub use log_softmax::{
    log_softmax, log_softmax_backward, quantized_log_softmax,
    quantized_log_softmax_backward, LogSoftmaxOp, QuantizedLogSoftmaxOp,
};
pub use relu::{
    quantized_relu, quantized_relu_backward, relu, relu_backward,
    QuantizedReluOp, ReluOp,
};
pub use relu6::{
    quantized_relu6, quantized_relu6_backward, relu6, relu6_backward,
    QuantizedRelu6Op, Relu6Op,
};
pub use sigmoid::{
    quantized_sigmoid, quantized_sigmoid_backward, sigmoid, sigmoid_backward,
    QuantizedSigmoidOp, SigmoidOp,
};
pub use silu::{
    quantized_silu, quantized_silu_backward, silu, silu_backward,
    QuantizedSiluOp, SiluOp,
};
pub use softmax::{
    quantized_softmax, quantized_softmax_backward, softmax, softmax_backward,
    QuantizedSoftmaxOp, SoftmaxOp,
};
pub use tanh::{
    quantized_tanh, quantized_tanh_backward, tanh, tanh_backward,
    QuantizedTanhOp, TanhOp,
};
