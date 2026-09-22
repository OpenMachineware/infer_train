use infer_train::quant::types::{BlockQ4_0, BlockQ4_1, BlockQ8_0, BlockQ8_1, QK4_0};
use infer_train::quant::vec_dot::scalar::q4_0::vec_dot_q4_0_q8_0;
use infer_train::quant::vec_dot::scalar::q8_0::vec_dot_q8_0_q8_0;
use infer_train::quant::vec_dot::scalar::q4_1::vec_dot_q4_1_q8_1;
use half::f16;

fn create_test_block_q4_0(values: &[i8]) -> BlockQ4_0 {
    assert_eq!(values.len(), QK4_0);
    let d = 1.0f32;
    let mut qs = [0u8; 16];

    for i in 0..16 {
        let lo = (values[i * 2] + 8) as u8;
        let hi = (values[i * 2 + 1] + 8) as u8;
        qs[i] = lo | (hi << 4);
    }

    BlockQ4_0 {
        d: f16::from_f32(d).to_bits(),
        qs,
    }
}

fn create_test_block_q8_0(values: &[i8]) -> BlockQ8_0 {
    assert_eq!(values.len(), QK4_0);
    let d = 1.0f32;
    BlockQ8_0 {
        d: f16::from_f32(d).to_bits(),
        qs: values.try_into().unwrap(),
    }
}

fn create_test_block_q4_1(values: &[i8], min: f32) -> BlockQ4_1 {
    assert_eq!(values.len(), QK4_0);
    let d = 1.0f32;
    let mut qs = [0u8; 16];

    for i in 0..16 {
        qs[i] = values[i * 2] as u8 | ((values[i * 2 + 1] as u8) << 4);
    }

    BlockQ4_1 {
        d: f16::from_f32(d).to_bits(),
        m: f16::from_f32(min).to_bits(),
        qs,
    }
}

fn create_test_block_q8_1(values: &[i8], sum: f32) -> BlockQ8_1 {
    assert_eq!(values.len(), QK4_0);
    let d = 1.0f32;
    BlockQ8_1 {
        d: f16::from_f32(d).to_bits(),
        s: f16::from_f32(sum).to_bits(),
        qs: values.try_into().unwrap(),
    }
}

#[test]
fn test_vec_dot_q4_0_q8_0() {
    // Test simple case
    let q4_values: Vec<i8> = (0..32).map(|i| if i < 16 { i as i8 - 8 } else { (31 - i) as i8 - 8 }).collect();
    let q8_values: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

    let x = vec![create_test_block_q4_0(&q4_values)];
    let y = vec![create_test_block_q8_0(&q8_values)];

    let result = vec_dot_q4_0_q8_0(QK4_0, &x, &y);

    // Compute expected result manually
    let expected: f32 = q4_values.iter().zip(q8_values.iter())
        .map(|(&a, &b)| a as f32 * b as f32)
        .sum();

    assert!((result - expected).abs() < 0.001,
        "Q4_0 × Q8_0: expected {}, got {}", expected, result);
}

#[test]
fn test_vec_dot_q8_0_q8_0() {
    let a_values: Vec<i8> = (0..32).map(|i| ((i % 7) - 3) as i8).collect();
    let b_values: Vec<i8> = (0..32).map(|i| ((i % 5) - 2) as i8).collect();

    let x = vec![create_test_block_q8_0(&a_values)];
    let y = vec![create_test_block_q8_0(&b_values)];

    let result = vec_dot_q8_0_q8_0(QK4_0, &x, &y);

    let expected: f32 = a_values.iter().zip(b_values.iter())
        .map(|(&a, &b)| a as f32 * b as f32)
        .sum();

    assert!((result - expected).abs() < 0.001,
        "Q8_0 × Q8_0: expected {}, got {}", expected, result);
}

#[test]
fn test_vec_dot_q4_1_q8_1() {
    // Q4_1: values 0-15, min = 0, so no offset
    let q4_values: Vec<i8> = (0..32).map(|i| (i % 16) as i8).collect();
    let q8_values: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
    let q8_sum: f32 = q8_values.iter().map(|&v| v as f32).sum();

    let x = vec![create_test_block_q4_1(&q4_values, 0.0)];
    let y = vec![create_test_block_q8_1(&q8_values, q8_sum)];

    let result = vec_dot_q4_1_q8_1(QK4_0, &x, &y);

    // Expected: sum of q4 * q8
    let expected: f32 = q4_values.iter().zip(q8_values.iter())
        .map(|(&a, &b)| a as f32 * b as f32)
        .sum();

    assert!((result - expected).abs() < 0.01,
        "Q4_1 × Q8_1: expected {}, got {}", expected, result);
}
