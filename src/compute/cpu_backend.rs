// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// CPU backend implementation

use std::sync::Arc;

use super::backend::{Backend, BackendError, BackendResult};
use super::device::Device;
use super::graph::ComputeGraph;
use super::op::{OpNode, OpParams, OpType};
use super::storage::{CpuStorage, StorageEnum};
use super::tensor::Tensor;
use crate::model::tensor::DataType;

/// CPU backend using NEON or scalar operations
#[derive(Debug)]
pub struct CpuBackend {
    _phantom: (),
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { _phantom: () }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn alloc_tensor(
        &self,
        name: &str,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> BackendResult<Tensor> {
        let n_elems: usize = shape.iter().product();
        let n_bytes = dtype.bytes_for_elems(n_elems);
        let storage =
            Arc::new(StorageEnum::Cpu(Arc::new(CpuStorage::zeros(n_bytes))));
        Ok(Tensor::new(name, shape, dtype, storage))
    }

    fn fill_tensor(
        &self,
        tensor: &mut Tensor,
        value: f32,
    ) -> BackendResult<()> {
        if tensor.dtype == DataType::F32 {
            if let Some(data) = tensor.as_f32_mut() {
                data.fill(value);
                return Ok(());
            }
        }

        // For other dtypes, create new storage with zeros
        if value == 0.0 {
            let n_bytes = tensor.byte_count();
            let new_storage = std::sync::Arc::new(StorageEnum::Cpu(
                std::sync::Arc::new(CpuStorage::zeros(n_bytes)),
            ));
            tensor.storage = new_storage;
            return Ok(());
        }

        Err(BackendError::ExecFailed(
            "Cannot fill non-CPU tensor or non-F32 dtype".to_string(),
        ))
    }

    fn copy_to_tensor(
        &self,
        tensor: &mut Tensor,
        data: &[u8],
    ) -> BackendResult<()> {
        let expected_size = tensor.byte_count();
        if data.len() != expected_size {
            return Err(BackendError::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![data.len()],
            });
        }

        // Create new storage with the data
        let new_storage = Arc::new(StorageEnum::Cpu(Arc::new(
            CpuStorage::new(data.to_vec()),
        )));
        tensor.storage = new_storage;
        Ok(())
    }

    fn copy_from_tensor(
        &self,
        tensor: &Tensor,
        data: &mut [u8],
    ) -> BackendResult<()> {
        let expected_size = tensor.byte_count();
        if data.len() != expected_size {
            return Err(BackendError::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![data.len()],
            });
        }

        if let Some(bytes) = tensor.as_bytes() {
            data.copy_from_slice(bytes);
            Ok(())
        } else {
            Err(BackendError::DeviceMismatch {
                expected: Device::Cpu,
                actual: tensor.device(),
            })
        }
    }

    fn execute(&self, graph: &mut ComputeGraph) -> BackendResult<()> {
        // Validate inputs
        for name in &graph.inputs {
            if !graph.tensors.contains_key(name) {
                return Err(BackendError::TensorNotFound(name.clone()));
            }
        }

        // Execute each operation in order
        let nodes: Vec<OpNode> = graph.nodes.clone();
        for node in &nodes {
            self.execute_op(node, graph)?;
        }

        Ok(())
    }

    fn synchronize(&self) -> BackendResult<()> {
        // CPU is synchronous, nothing to do
        Ok(())
    }

    fn supports_op(&self, op_type: &OpType) -> bool {
        matches!(
            op_type,
            OpType::Add
                | OpType::Mul
                | OpType::Scale
                | OpType::RMSNorm
                | OpType::Silu
                | OpType::Softmax
                | OpType::MatVec
        )
    }

    fn supports_dtype(&self, dtype: &DataType) -> bool {
        // CPU supports all dtypes through dequantization
        matches!(
            dtype,
            DataType::F32
                | DataType::F16
                | DataType::BF16
                | DataType::Q4_0
                | DataType::Q4_1
                | DataType::Q5_0
                | DataType::Q5_1
                | DataType::Q8_0
        )
    }
}

impl CpuBackend {
    /// Execute a single operation
    fn execute_op(
        &self,
        node: &OpNode,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // Get input data first to avoid borrow conflicts
        let inputs_data: Vec<(String, Vec<usize>, DataType, Vec<u8>)> = node
            .inputs
            .iter()
            .map(|name| {
                let tensor = graph.tensors.get(name).ok_or_else(|| {
                    BackendError::TensorNotFound(name.clone())
                })?;
                let data = tensor
                    .as_bytes()
                    .ok_or_else(|| BackendError::DeviceMismatch {
                        expected: Device::Cpu,
                        actual: tensor.device(),
                    })?
                    .to_vec();
                Ok((
                    tensor.name.clone(),
                    tensor.shape.clone(),
                    tensor.dtype,
                    data,
                ))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        match node.op_type {
            OpType::RMSNorm => self.op_rms_norm(node, inputs_data, graph),
            OpType::Silu => self.op_silu(node, inputs_data, graph),
            OpType::Softmax => self.op_softmax(node, inputs_data, graph),
            OpType::MatVec => self.op_matvec(node, inputs_data, graph),
            OpType::Add => self.op_add(node, inputs_data, graph),
            OpType::Mul => self.op_mul(node, inputs_data, graph),
            OpType::Scale => self.op_scale(node, inputs_data, graph),
            _ => Err(BackendError::OpNotSupported(format!("{}", node.op_type))),
        }
    }

    fn op_rms_norm(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, Vec<u8>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let input = &inputs[0];
        let weight = &inputs[1];
        let eps = node.params.eps.unwrap_or(1e-5);

        if input.2 != DataType::F32 {
            return Err(BackendError::DtypeMismatch {
                expected: DataType::F32,
                actual: input.2,
            });
        }

        let input_data: &[f32] = bytemuck::cast_slice(&input.3);
        let weight_data: &[f32] = bytemuck::cast_slice(&weight.3);

        let hidden_size = input.1[0];
        let n_tokens = input.1.get(1).copied().unwrap_or(1);

        let mut output_f32 = vec![0.0f32; hidden_size * n_tokens];

        for t in 0..n_tokens {
            let offset = t * hidden_size;
            let row = &input_data[offset..offset + hidden_size];

            let ss: f32 = row.iter().map(|&x| x * x).sum();
            let rms = (ss / hidden_size as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for i in 0..hidden_size {
                output_f32[offset + i] = row[i] * inv_rms * weight_data[i];
            }
        }

        let output =
            Tensor::from_f32(&node.output, input.1.clone(), output_f32);
        graph.add_tensor(output);
        Ok(())
    }

    fn op_silu(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, Vec<u8>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let input = &inputs[0];

        if input.2 != DataType::F32 {
            return Err(BackendError::DtypeMismatch {
                expected: DataType::F32,
                actual: input.2,
            });
        }

        let input_data: &[f32] = bytemuck::cast_slice(&input.3);
        let mut output_f32 = vec![0.0f32; input_data.len()];

        for (i, &x) in input_data.iter().enumerate() {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            output_f32[i] = x * sigmoid;
        }

        let output =
            Tensor::from_f32(&node.output, input.1.clone(), output_f32);
        graph.add_tensor(output);
        Ok(())
    }

    fn op_softmax(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, Vec<u8>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let input = &inputs[0];

        if input.2 != DataType::F32 {
            return Err(BackendError::DtypeMismatch {
                expected: DataType::F32,
                actual: input.2,
            });
        }

        let input_data: &[f32] = bytemuck::cast_slice(&input.3);
        let mut output_f32 = vec![0.0f32; input_data.len()];

        let dim0 = input.1[0];
        let dim1 = input.1.get(1).copied().unwrap_or(1);

        for j in 0..dim1 {
            let offset = j * dim0;
            let row = &input_data[offset..offset + dim0];

            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let mut sum = 0.0;
            for (i, &x) in row.iter().enumerate() {
                let exp_val = (x - max_val).exp();
                output_f32[offset + i] = exp_val;
                sum += exp_val;
            }

            let inv_sum = 1.0 / sum;
            for i in 0..dim0 {
                output_f32[offset + i] *= inv_sum;
            }
        }

        let output =
            Tensor::from_f32(&node.output, input.1.clone(), output_f32);
        graph.add_tensor(output);
        Ok(())
    }

    fn op_matvec(
        &self,
        _node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, Vec<u8>)>,
        _graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let weight = &inputs[0];
        let _vec = &inputs[1];

        // For quantized weights, use existing vec_dot
        if weight.2.is_quantized() {
            // TODO: Implement using quant::vec_dot
            return Err(BackendError::OpNotSupported(
                "Quantized matvec not yet implemented".to_string(),
            ));
        }

        // TODO: Implement F32 matvec
        Err(BackendError::OpNotSupported(
            "F32 matvec not yet implemented".to_string(),
        ))
    }

    fn op_add(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, Vec<u8>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let a = &inputs[0];
        let b = &inputs[1];

        if a.2 != DataType::F32 || b.2 != DataType::F32 {
            return Err(BackendError::DtypeMismatch {
                expected: DataType::F32,
                actual: a.2,
            });
        }

        let a_data: &[f32] = bytemuck::cast_slice(&a.3);
        let b_data: &[f32] = bytemuck::cast_slice(&b.3);

        let mut output_f32 = vec![0.0f32; a_data.len()];
        for (i, (&x, &y)) in a_data.iter().zip(b_data.iter()).enumerate() {
            output_f32[i] = x + y;
        }

        let output = Tensor::from_f32(&node.output, a.1.clone(), output_f32);
        graph.add_tensor(output);
        Ok(())
    }

    fn op_mul(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, Vec<u8>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let a = &inputs[0];
        let b = &inputs[1];

        if a.2 != DataType::F32 || b.2 != DataType::F32 {
            return Err(BackendError::DtypeMismatch {
                expected: DataType::F32,
                actual: a.2,
            });
        }

        let a_data: &[f32] = bytemuck::cast_slice(&a.3);
        let b_data: &[f32] = bytemuck::cast_slice(&b.3);

        let mut output_f32 = vec![0.0f32; a_data.len()];
        for (i, (&x, &y)) in a_data.iter().zip(b_data.iter()).enumerate() {
            output_f32[i] = x * y;
        }

        let output = Tensor::from_f32(&node.output, a.1.clone(), output_f32);
        graph.add_tensor(output);
        Ok(())
    }

    fn op_scale(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, Vec<u8>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let input = &inputs[0];
        let scale = node.params.scale.unwrap_or(1.0);

        if input.2 != DataType::F32 {
            return Err(BackendError::DtypeMismatch {
                expected: DataType::F32,
                actual: input.2,
            });
        }

        let input_data: &[f32] = bytemuck::cast_slice(&input.3);
        let output_f32: Vec<f32> =
            input_data.iter().map(|&x| x * scale).collect();

        let output =
            Tensor::from_f32(&node.output, input.1.clone(), output_f32);
        graph.add_tensor(output);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_rms_norm() {
        let backend = CpuBackend::new();

        // Create input tensor
        let input =
            Tensor::from_f32("input", vec![4, 1], vec![1.0, 2.0, 3.0, 4.0]);

        // Create weight tensor
        let weight =
            Tensor::from_f32("weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]);

        // Create graph
        let mut graph = ComputeGraph::new();
        graph.add_tensor(input.clone());
        graph.add_tensor(weight.clone());
        graph.add_input("input");
        graph.op(
            OpType::RMSNorm,
            vec!["input".to_string(), "weight".to_string()],
            "output".to_string(),
        );

        // Execute
        backend.execute(&mut graph).unwrap();

        // Check output
        let output = graph.get_tensor("output").unwrap();
        let output_data = output.as_f32().unwrap();

        // RMS = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + 1e-5) ≈ 2.738
        // normalized = [0.365, 0.730, 1.095, 1.460]
        assert!((output_data[0] - 0.365).abs() < 0.01);
    }

    #[test]
    fn test_cpu_backend_silu() {
        let backend = CpuBackend::new();

        let input = Tensor::from_f32("input", vec![3], vec![0.0, 1.0, -1.0]);

        let mut graph = ComputeGraph::new();
        graph.add_tensor(input);
        graph.op(OpType::Silu, vec!["input".to_string()], "output".to_string());

        backend.execute(&mut graph).unwrap();

        let output = graph.get_tensor("output").unwrap();
        let output_data = output.as_f32().unwrap();

        // SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(-1) ≈ -0.269
        assert!((output_data[0] - 0.0).abs() < 0.01);
        assert!((output_data[1] - 0.731).abs() < 0.01);
        assert!((output_data[2] - (-0.269)).abs() < 0.01);
    }
}
