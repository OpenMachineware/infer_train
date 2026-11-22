// Input quantization types for inference

use crate::backend::arm::quant;
use crate::types::*;

/// Quantized input storage with pre-allocated buffer
pub struct QuantizedInput {
    /// Q8_0 format blocks (for Q4_0, Q5_0, Q8_0 weights)
    pub q8_0: Vec<BlockQ8_0>,
    /// Q8_K format blocks (for Q4_K, Q5_K, Q6_K, Q8_K, IQ4_XS weights)
    pub q8_k: Vec<BlockQ8_K>,
    /// Current active format
    pub active: QuantFormat,
}

/// Batch quantized input for prefill: [n_tokens * n_blocks]
pub struct QuantizedInputBatch {
    pub q8_0: Vec<BlockQ8_0>,
    pub q8_k: Vec<BlockQ8_K>,
    pub active: QuantFormat,
    pub n_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantFormat {
    None,
    Q8_0,
    Q8_K,
}

impl QuantizedInput {
    pub fn new() -> Self {
        Self { q8_0: Vec::new(), q8_k: Vec::new(), active: QuantFormat::None }
    }

    pub fn resize(&mut self, in_features: usize, format: QuantFormat) {
        match format {
            QuantFormat::Q8_0 => {
                let nb = (in_features + QK8_0 - 1) / QK8_0;
                if self.q8_0.len() != nb {
                    self.q8_0.resize(nb, BlockQ8_0 { d: 0, qs: [0; QK8_0] });
                }
            }
            QuantFormat::Q8_K => {
                let nb = (in_features + QK_K - 1) / QK_K;
                if self.q8_k.len() != nb {
                    self.q8_k.resize(nb, unsafe { std::mem::zeroed() });
                }
            }
            QuantFormat::None => {}
        }
        self.active = format;
    }
}

impl QuantizedInputBatch {
    pub fn new() -> Self {
        Self {
            q8_0: Vec::new(),
            q8_k: Vec::new(),
            active: QuantFormat::None,
            n_tokens: 0,
        }
    }

    /// Resize for batch processing: n_tokens * in_features
    pub fn resize(
        &mut self,
        n_tokens: usize,
        in_features: usize,
        format: QuantFormat,
    ) {
        self.n_tokens = n_tokens;
        match format {
            QuantFormat::Q8_0 => {
                let nb = n_tokens * ((in_features + QK8_0 - 1) / QK8_0);
                if self.q8_0.len() != nb {
                    self.q8_0.resize(nb, BlockQ8_0 { d: 0, qs: [0; QK8_0] });
                }
            }
            QuantFormat::Q8_K => {
                let nb = n_tokens * ((in_features + QK_K - 1) / QK_K);
                if self.q8_k.len() != nb {
                    self.q8_k.resize(nb, unsafe { std::mem::zeroed() });
                }
            }
            QuantFormat::None => {}
        }
        self.active = format;
    }
}

/// Quantize input into the provided buffer
pub fn quantize_input_to(
    input: &[f32],
    quant_type: QuantType,
    quant_buf: &mut QuantizedInput,
) -> Result<QuantFormat, String> {
    match quant_type {
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            quant_buf.resize(input.len(), QuantFormat::Q8_0);
            quant::quantize_q8_0(input, &mut quant_buf.q8_0);
            Ok(QuantFormat::Q8_0)
        }
        QuantType::Q4_K
        | QuantType::Q5_K
        | QuantType::Q6_K
        | QuantType::Q8_K
        | QuantType::IQ4_XS => {
            quant_buf.resize(input.len(), QuantFormat::Q8_K);
            quant::quantize_q8_k(input, &mut quant_buf.q8_k);
            Ok(QuantFormat::Q8_K)
        }
        _ => Err(format!("Unsupported quant type: {:?}", quant_type)),
    }
}

/// Batch quantize input: [n_tokens * hidden_size] -> QuantizedInputBatch
pub fn quantize_input_batch(
    input: &[f32],
    hidden_size: usize,
    n_tokens: usize,
    quant_type: QuantType,
    quant_buf: &mut QuantizedInputBatch,
) -> Result<QuantFormat, String> {
    match quant_type {
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            quant_buf.resize(n_tokens, hidden_size, QuantFormat::Q8_0);
            let nb_per_token = (hidden_size + QK8_0 - 1) / QK8_0;
            for token_idx in 0..n_tokens {
                let in_start = token_idx * hidden_size;
                let out_start = token_idx * nb_per_token;
                quant::quantize_q8_0(
                    &input[in_start..in_start + hidden_size],
                    &mut quant_buf.q8_0[out_start..out_start + nb_per_token],
                );
            }
            Ok(QuantFormat::Q8_0)
        }
        QuantType::Q4_K
        | QuantType::Q5_K
        | QuantType::Q6_K
        | QuantType::Q8_K
        | QuantType::IQ4_XS => {
            quant_buf.resize(n_tokens, hidden_size, QuantFormat::Q8_K);
            let nb_per_token = (hidden_size + QK_K - 1) / QK_K;
            for token_idx in 0..n_tokens {
                let in_start = token_idx * hidden_size;
                let out_start = token_idx * nb_per_token;
                quant::quantize_q8_k(
                    &input[in_start..in_start + hidden_size],
                    &mut quant_buf.q8_k[out_start..out_start + nb_per_token],
                );
            }
            Ok(QuantFormat::Q8_K)
        }
        _ => Err(format!("Unsupported quant type for batch: {:?}", quant_type)),
    }
}
