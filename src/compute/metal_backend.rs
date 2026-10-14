// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Metal backend implementation using existing GPU kernels

use metal::{Buffer, MTLResourceOptions};

use super::backend::{Backend, BackendError, BackendResult};
use super::device::Device;
use super::graph::ComputeGraph;
use super::op::{OpNode, OpType};
use super::tensor::Tensor;
use crate::model::tensor::DataType;
use crate::quant::vec_dot::gpu::metal::MetalContext;

/// Metal backend using existing GPU kernels from MetalContext
#[derive(Debug)]
pub struct MetalBackend {
    ctx: MetalContext,
}

impl MetalBackend {
    pub fn new() -> BackendResult<Self> {
        let ctx = MetalContext::new().map_err(BackendError::AllocFailed)?;
        Ok(Self { ctx })
    }

    /// Create a Metal buffer
    pub fn create_buffer(&self, size: usize) -> std::rc::Rc<Buffer> {
        std::rc::Rc::new(
            self.ctx
                .device()
                .new_buffer(size as u64, MTLResourceOptions::StorageModeShared),
        )
    }

    /// Create a Metal buffer from data
    pub fn create_buffer_with_data(&self, data: &[u8]) -> std::rc::Rc<Buffer> {
        std::rc::Rc::new(self.ctx.device().new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    /// Get reference to the MetalContext
    pub fn context(&self) -> &MetalContext {
        &self.ctx
    }
}

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        Device::Metal
    }

    fn alloc_tensor(
        &self,
        name: &str,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> BackendResult<Tensor> {
        let n_bytes = dtype.bytes_for_elems(shape.iter().product::<usize>());
        let buffer = self.create_buffer(n_bytes);
        Ok(Tensor::from_metal_buffer(name, shape, dtype, buffer))
    }

    fn fill_tensor(
        &self,
        tensor: &mut Tensor,
        value: f32,
    ) -> BackendResult<()> {
        let buffer = tensor.as_metal_buffer().ok_or_else(|| {
            BackendError::DeviceMismatch {
                expected: Device::Metal,
                actual: tensor.device(),
            }
        })?;

        let contents = buffer.contents();
        let data = unsafe {
            std::slice::from_raw_parts_mut(
                contents as *mut u8,
                tensor.byte_count(),
            )
        };

        if tensor.dtype == DataType::F32 {
            let f32_data = bytemuck::cast_slice_mut::<u8, f32>(data);
            f32_data.fill(value);
        } else if value == 0.0 {
            data.fill(0);
        } else {
            return Err(BackendError::ExecFailed(
                "Can only fill Metal tensors with 0 or f32 values".to_string(),
            ));
        }

        Ok(())
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

        let buffer = tensor.as_metal_buffer().ok_or_else(|| {
            BackendError::DeviceMismatch {
                expected: Device::Metal,
                actual: tensor.device(),
            }
        })?;

        let contents = buffer.contents();
        let dest = unsafe {
            std::slice::from_raw_parts_mut(contents as *mut u8, expected_size)
        };
        dest.copy_from_slice(data);

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

        let buffer = tensor.as_metal_buffer().ok_or_else(|| {
            BackendError::DeviceMismatch {
                expected: Device::Metal,
                actual: tensor.device(),
            }
        })?;

        let contents = buffer.contents();
        let src = unsafe {
            std::slice::from_raw_parts(contents as *const u8, expected_size)
        };
        data.copy_from_slice(src);

        Ok(())
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
        self.ctx.device().new_command_queue().new_command_buffer().commit();
        Ok(())
    }

    fn supports_op(&self, op_type: &OpType) -> bool {
        matches!(
            op_type,
            OpType::RMSNorm
                | OpType::Softmax
                | OpType::Silu
                | OpType::Rope
                | OpType::MatVec
                | OpType::FlashAttn
                | OpType::Add
                | OpType::Mul
        )
    }

    fn supports_dtype(&self, _dtype: &DataType) -> bool {
        // Metal supports all quantized formats through MV kernels
        true
    }
}

impl MetalBackend {
    /// Execute a single operation
    fn execute_op(
        &self,
        node: &OpNode,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // Get input data first to avoid borrow conflicts
        let inputs_data: Vec<(
            String,
            Vec<usize>,
            DataType,
            std::rc::Rc<Buffer>,
        )> = node
            .inputs
            .iter()
            .map(|name| {
                let tensor = graph.tensors.get(name).ok_or_else(|| {
                    BackendError::TensorNotFound(name.clone())
                })?;
                let buffer = tensor.as_metal_buffer().ok_or_else(|| {
                    BackendError::DeviceMismatch {
                        expected: Device::Metal,
                        actual: tensor.device(),
                    }
                })?;
                Ok((
                    tensor.name.clone(),
                    tensor.shape.clone(),
                    tensor.dtype,
                    buffer,
                ))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        match node.op_type {
            OpType::RMSNorm => self.op_rms_norm(node, inputs_data, graph),
            OpType::Softmax => self.op_softmax(node, inputs_data, graph),
            OpType::Silu => self.op_silu(node, inputs_data, graph),
            OpType::Rope => self.op_rope(node, inputs_data, graph),
            OpType::MatVec => self.op_matvec(node, inputs_data, graph),
            OpType::FlashAttn => self.op_flash_attn(node, inputs_data, graph),
            OpType::Add => self.op_add(node, inputs_data, graph),
            OpType::Mul => self.op_mul(node, inputs_data, graph),
            _ => Err(BackendError::OpNotSupported(format!("{}", node.op_type))),
        }
    }

    fn op_rms_norm(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let input = &inputs[0];
        let weight = &inputs[1];
        let eps = node.params.eps.unwrap_or(1e-5);

        let n_rows = input.1.get(1).copied().unwrap_or(1);
        let n_cols = input.1[0];

        // Use GPU-native method (zero-copy)
        let output_buffer = self
            .ctx
            .rms_norm_f32_gpu(&input.3, &weight.3, n_rows, n_cols, eps)
            .map_err(BackendError::ExecFailed)?;

        // Create output tensor with the returned GPU buffer
        let output = Tensor::from_metal_buffer(
            &node.output,
            input.1.clone(),
            DataType::F32,
            output_buffer,
        );
        graph.add_tensor(output);

        Ok(())
    }

    fn op_softmax(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let input = &inputs[0];

        let ncols = input.1[0];
        let nrows = input.1.get(1).copied().unwrap_or(1);

        // Use GPU-native method (zero-copy)
        let output_buffer = self
            .ctx
            .softmax_f32_fast_gpu(&input.3, nrows, ncols)
            .map_err(BackendError::ExecFailed)?;

        // Create output tensor with the returned GPU buffer
        let output = Tensor::from_metal_buffer(
            &node.output,
            input.1.clone(),
            DataType::F32,
            output_buffer,
        );
        graph.add_tensor(output);

        Ok(())
    }

    fn op_silu(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let input = &inputs[0];
        let n = input.1.iter().product();

        // Use GPU-native method (zero-copy)
        let output_buffer = self
            .ctx
            .silu_f32_gpu(&input.3, n)
            .map_err(BackendError::ExecFailed)?;

        // Create output tensor with the returned GPU buffer
        let output = Tensor::from_metal_buffer(
            &node.output,
            input.1.clone(),
            DataType::F32,
            output_buffer,
        );
        graph.add_tensor(output);

        Ok(())
    }

    fn op_rope(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let input = &inputs[0];
        let pos = node.params.pos.unwrap_or(0);

        let n_dims = input.1[0];
        let n_heads = input.1.get(1).copied().unwrap_or(1);
        let n_seqs = input.1.get(2).copied().unwrap_or(1);
        let freq_base = node.params.freq_base.unwrap_or(10000.0);
        let freq_scale = node.params.freq_scale.unwrap_or(1.0);

        let pos_vec: Vec<i32> = vec![pos as i32; n_seqs];

        // Use GPU-native method (zero-copy)
        let output_buffer = self
            .ctx
            .rope_neox_f32_gpu(
                &input.3, &pos_vec, n_seqs, n_heads, n_dims, freq_base,
                freq_scale,
            )
            .map_err(BackendError::ExecFailed)?;

        // Create output tensor with the returned GPU buffer
        let output = Tensor::from_metal_buffer(
            &node.output,
            input.1.clone(),
            DataType::F32,
            output_buffer,
        );
        graph.add_tensor(output);

        Ok(())
    }

    fn op_matvec(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let weight = &inputs[0];
        let vec = &inputs[1];

        let m = weight.1[1]; // Output size (columns in column-major)
        let k = weight.1[0]; // Input size (rows in column-major)

        // Use GPU-native methods (fully zero-copy, both weights and input as GPU buffers)
        let output_buffer = match weight.2 {
            DataType::Q4_0 => self
                .ctx
                .mv_q4_0_f32_gpu_weights(m, k, &weight.3, &vec.3)
                .map_err(BackendError::ExecFailed)?,
            DataType::Q8_0 => self
                .ctx
                .mv_q8_0_f32_gpu_weights(m, k, &weight.3, &vec.3)
                .map_err(BackendError::ExecFailed)?,
            DataType::Q4_1 => self
                .ctx
                .mv_q4_1_f32_gpu_weights(m, k, &weight.3, &vec.3)
                .map_err(BackendError::ExecFailed)?,
            DataType::Q5_0 => self
                .ctx
                .mv_q5_0_f32_gpu_weights(m, k, &weight.3, &vec.3)
                .map_err(BackendError::ExecFailed)?,
            DataType::Q5_1 => self
                .ctx
                .mv_q5_1_f32_gpu_weights(m, k, &weight.3, &vec.3)
                .map_err(BackendError::ExecFailed)?,
            DataType::IQ4_NL => self
                .ctx
                .mv_iq4_nl_f32_gpu_weights(m, k, &weight.3, &vec.3)
                .map_err(BackendError::ExecFailed)?,
            _ => {
                return Err(BackendError::OpNotSupported(format!(
                    "MatVec for weight dtype {:?}",
                    weight.2
                )));
            }
        };

        // Create output tensor with the returned GPU buffer
        let output = Tensor::from_metal_buffer(
            &node.output,
            vec![m],
            DataType::F32,
            output_buffer,
        );
        graph.add_tensor(output);

        Ok(())
    }

    fn op_flash_attn(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // inputs: [q, k_cache, v_cache]
        let q = &inputs[0];
        let k_cache = &inputs[1];
        let v_cache = &inputs[2];

        // Get parameters from node
        let n_heads = node.params.n_heads.unwrap_or(1);
        let n_kv_heads = node.params.n_kv_heads.unwrap_or(n_heads);
        let head_dim = node.params.head_dim.unwrap_or(128);
        let seq_len = node.params.seq_len.unwrap_or(1);
        let n_seqs = q.1.get(2).copied().unwrap_or(1);

        // Use GPU-native method (zero-copy)
        let output_buffer = self
            .ctx
            .flash_attn_f16_gpu(
                n_seqs, n_heads, n_kv_heads, head_dim, seq_len, &q.3,
                &k_cache.3, &v_cache.3,
            )
            .map_err(BackendError::ExecFailed)?;

        // Create output tensor with the returned GPU buffer
        let output = Tensor::from_metal_buffer(
            &node.output,
            vec![head_dim, n_heads, n_seqs],
            DataType::F32,
            output_buffer,
        );
        graph.add_tensor(output);

        Ok(())
    }

    fn op_add(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let a = &inputs[0];
        let b = &inputs[1];
        let n = a.1.iter().product();

        // Use GPU-native method (zero-copy)
        let output_buffer = self
            .ctx
            .add_f32_gpu(&a.3, &b.3, n)
            .map_err(BackendError::ExecFailed)?;

        // Create output tensor with the returned GPU buffer
        let output = Tensor::from_metal_buffer(
            &node.output,
            a.1.clone(),
            DataType::F32,
            output_buffer,
        );
        graph.add_tensor(output);

        Ok(())
    }

    fn op_mul(
        &self,
        node: &OpNode,
        inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        let a = &inputs[0];
        let b = &inputs[1];
        let n = a.1.iter().product();

        // Use GPU-native method (zero-copy)
        let output_buffer = self
            .ctx
            .mul_f32_gpu(&a.3, &b.3, n)
            .map_err(BackendError::ExecFailed)?;

        // Create output tensor with the returned GPU buffer
        let output = Tensor::from_metal_buffer(
            &node.output,
            a.1.clone(),
            DataType::F32,
            output_buffer,
        );
        graph.add_tensor(output);

        Ok(())
    }
}
