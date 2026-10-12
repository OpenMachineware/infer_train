// Metal GPU operators for Apple Silicon
// Re-exports Metal GPU implementations

pub use crate::ops::gpu::metal::{
    ActivationMetalContext, FlashAttnMetalContext, RmsNormMetalContext,
    RopeMetalContext, SoftmaxMetalContext,
};
