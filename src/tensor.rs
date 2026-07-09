// src/tensor.rs

use crate::dtype::DType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: DType> {
    data: Vec<T>,
    shape: Vec<usize>,
    #[serde(skip)]
    #[allow(dead_code)]
    stride: Vec<usize>,
    scale: Option<f32>,
    zero_point: Option<f32>,
}

impl<T: DType> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &[usize]) -> Self {
        let stride = Self::compute_stride(shape);
        Tensor {
            data,
            shape: shape.to_vec(),
            stride,
            scale: None,
            zero_point: None,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![T::zero(); size];
        Tensor::new(data, shape)
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![T::one(); size];
        Tensor::new(data, shape)
    }

    fn compute_stride(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return vec![1]; // stride for empty shape is [1]
        }
        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        stride
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn num_elements(&self) -> usize {
        self.data.len()
    }
    pub fn is_quantized(&self) -> bool {
        self.scale.is_some()
    }
    pub fn scale(&self) -> Option<f32> {
        self.scale
    }
    pub fn zero_point(&self) -> Option<f32> {
        self.zero_point
    }

    pub fn dequantize(&self) -> Option<Tensor<f32>>
    where
        T: Into<f32> + Copy,
    {
        if !self.is_quantized() {
            return None;
        }
        let scale = self.scale.unwrap();
        let zero_point = self.zero_point.unwrap();
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| (x.into() - zero_point) * scale)
            .collect();
        Some(Tensor::new(data, &self.shape))
    }
}

// Quantized Tensor constructor for i8
impl Tensor<i8> {
    pub fn new_quantized(
        data: Vec<i8>,
        shape: &[usize],
        scale: f32,
        zero_point: f32,
    ) -> Self {
        let stride = Self::compute_stride(shape);
        Tensor {
            data,
            shape: shape.to_vec(),
            stride,
            scale: Some(scale),
            zero_point: Some(zero_point),
        }
    }
}

impl<T: DType> From<(Vec<T>, &[usize])> for Tensor<T> {
    fn from((data, shape): (Vec<T>, &[usize])) -> Self {
        Tensor::new(data, shape)
    }
}
