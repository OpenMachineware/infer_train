// FP16 lookup tables for activation functions
// Matches llama.cpp's ggml_table_gelu_f16 and ggml_table_gelu_quick_f16
// Tables are initialized at runtime, same as llama.cpp

use half::f16;
use std::sync::OnceLock;

const GELU_COEF_A: f64 = 0.044715;
const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
const GELU_QUICK_COEF: f64 = -1.702;

// GELU table: 128 KB (65536 * 2 bytes)
static GELU_TABLE: OnceLock<Vec<f16>> = OnceLock::new();

// GELU-Quick table: 128 KB
static GELU_QUICK_TABLE: OnceLock<Vec<f16>> = OnceLock::new();

fn init_gelu_table() -> Vec<f16> {
    let mut table = vec![f16::ZERO; 65536];

    for i in 0..=65535u16 {
        let f = f16::from_bits(i).to_f32() as f64;

        let inner = 1.0 + GELU_COEF_A * f * f;
        let tanh_arg = SQRT_2_OVER_PI * f * inner;
        let result = 0.5 * f * (1.0 + tanh_arg.tanh());

        table[i as usize] = f16::from_f32(result as f32);
    }

    table
}

fn init_gelu_quick_table() -> Vec<f16> {
    let mut table = vec![f16::ZERO; 65536];

    for i in 0..=65535u16 {
        let f = f16::from_bits(i).to_f32() as f64;

        let result = f / (1.0 + (GELU_QUICK_COEF * f).exp());

        table[i as usize] = f16::from_f32(result as f32);
    }

    table
}

fn get_gelu_table() -> &'static Vec<f16> {
    GELU_TABLE.get_or_init(init_gelu_table)
}

fn get_gelu_quick_table() -> &'static Vec<f16> {
    GELU_QUICK_TABLE.get_or_init(init_gelu_quick_table)
}

/// GELU using FP16 lookup table
pub fn gelu_f32_table(x: &[f32], dst: &mut [f32]) {
    let table = get_gelu_table();

    for i in 0..x.len() {
        let f16_val = f16::from_f32(x[i]);
        let index = f16_val.to_bits() as usize;
        dst[i] = table[index].to_f32();
    }
}

/// GELU for F16 input - direct table lookup (no conversion, matches llama.cpp)
pub fn gelu_f16_table(x: &[f16], dst: &mut [f16]) {
    let table = get_gelu_table();
    let n = x.len().min(dst.len());

    for i in 0..n {
        let index = x[i].to_bits() as usize;
        dst[i] = table[index];
    }
}

/// GELU-Quick using FP16 lookup table
pub fn gelu_quick_f32_table(x: &[f32], dst: &mut [f32]) {
    let table = get_gelu_quick_table();

    for i in 0..x.len() {
        let f16_val = f16::from_f32(x[i]);
        let index = f16_val.to_bits() as usize;
        dst[i] = table[index].to_f32();
    }
}

/// GELU-Quick for F16 input - direct table lookup (no conversion, matches llama.cpp)
pub fn gelu_quick_f16_table(x: &[f16], dst: &mut [f16]) {
    let table = get_gelu_quick_table();
    let n = x.len().min(dst.len());

    for i in 0..n {
        let index = x[i].to_bits() as usize;
        dst[i] = table[index];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu_table() {
        let test_values: Vec<f32> =
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0f32; test_values.len()];

        gelu_f32_table(&test_values, &mut dst);

        for i in 0..test_values.len() {
            assert!(dst[i].is_finite());
        }
    }

    #[test]
    fn test_gelu_quick_table() {
        let test_values: Vec<f32> =
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0f32; test_values.len()];

        gelu_quick_f32_table(&test_values, &mut dst);

        for i in 0..test_values.len() {
            assert!(dst[i].is_finite());
        }
    }
}
