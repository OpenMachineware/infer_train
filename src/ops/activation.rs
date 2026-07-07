// src/ops/activation/mod.rs

pub mod relu;
pub mod gelu;
pub mod silu;
pub mod sigmoid;
pub mod tanh;
pub mod leaky_relu;
pub mod elu;
pub mod hard_swish;
pub mod softmax;
pub mod log_softmax;
pub mod relu6;

pub use relu::{relu, relu_backward, quantized_relu, quantized_relu_backward, ReluOp, QuantizedReluOp};
pub use relu6::{relu6, relu6_backward, quantized_relu6, quantized_relu6_backward, Relu6Op, QuantizedRelu6Op};
pub use gelu::{gelu, gelu_backward, quantized_gelu, quantized_gelu_backward, GeluOp, QuantizedGeluOp};
pub use silu::{silu, silu_backward, quantized_silu, quantized_silu_backward, SiluOp, QuantizedSiluOp};
pub use sigmoid::{sigmoid, sigmoid_backward, quantized_sigmoid, quantized_sigmoid_backward, SigmoidOp, QuantizedSigmoidOp};
pub use tanh::{tanh, tanh_backward, quantized_tanh, quantized_tanh_backward, TanhOp, QuantizedTanhOp};
pub use leaky_relu::{leaky_relu, leaky_relu_backward, quantized_leaky_relu, quantized_leaky_relu_backward, LeakyReluOp, QuantizedLeakyReluOp};
pub use elu::{elu, elu_backward, quantized_elu, quantized_elu_backward, EluOp, QuantizedEluOp};
pub use hard_swish::{hard_swish, hard_swish_backward, quantized_hard_swish, quantized_hard_swish_backward, HardSwishOp, QuantizedHardSwishOp};
pub use softmax::{softmax, softmax_backward, quantized_softmax, quantized_softmax_backward, SoftmaxOp, QuantizedSoftmaxOp};
pub use log_softmax::{log_softmax, log_softmax_backward, quantized_log_softmax, quantized_log_softmax_backward, LogSoftmaxOp, QuantizedLogSoftmaxOp};
