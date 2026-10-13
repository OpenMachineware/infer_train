// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Metal backend implementation

use metal::{Buffer, Device as MetalDevice, MTLResourceOptions};

use super::backend::{Backend, BackendError, BackendResult};
use super::device::Device;
use super::graph::ComputeGraph;
use super::op::{OpNode, OpType};
use super::tensor::Tensor;
use crate::model::tensor::DataType;

/// Metal backend using existing GPU kernels
#[derive(Debug)]
pub struct MetalBackend {
    device: MetalDevice,
    command_queue: metal::CommandQueue,
}

impl MetalBackend {
    pub fn new() -> BackendResult<Self> {
        let device = MetalDevice::system_default().ok_or_else(|| {
            BackendError::AllocFailed("No Metal device found".to_string())
        })?;

        let command_queue = device.new_command_queue();

        Ok(Self { device, command_queue })
    }

    /// Create a Metal buffer
    pub fn create_buffer(&self, size: usize) -> std::rc::Rc<Buffer> {
        let buffer = self
            .device
            .new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        std::rc::Rc::new(buffer)
    }

    /// Create a Metal buffer from data
    pub fn create_buffer_with_data(&self, data: &[u8]) -> std::rc::Rc<Buffer> {
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        std::rc::Rc::new(buffer)
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
        } else {
            // For other dtypes, fill with zeros
            if value == 0.0 {
                data.fill(0);
            } else {
                return Err(BackendError::ExecFailed(
                    "Can only fill Metal tensors with 0 or f32 values"
                        .to_string(),
                ));
            }
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
        let cmd_buffer = self.command_queue.new_command_buffer();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
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
        )
    }

    fn supports_dtype(&self, dtype: &DataType) -> bool {
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

        // Get data from Metal buffers
        let input_contents = input.3.contents();
        let input_f32: &[f32] = bytemuck::cast_slice(unsafe {
            std::slice::from_raw_parts(
                input_contents as *const u8,
                input.1.iter().product::<usize>() * 4,
            )
        });

        let weight_contents = weight.3.contents();
        let weight_f32: &[f32] = bytemuck::cast_slice(unsafe {
            std::slice::from_raw_parts(
                weight_contents as *const u8,
                weight.1.iter().product::<usize>() * 4,
            )
        });

        let hidden_size = input.1[0];
        let n_tokens = input.1.get(1).copied().unwrap_or(1);

        let mut output_f32 = vec![0.0f32; hidden_size * n_tokens];

        for t in 0..n_tokens {
            let offset = t * hidden_size;
            let row = &input_f32[offset..offset + hidden_size];

            let ss: f32 = row.iter().map(|&x| x * x).sum();
            let rms = (ss / hidden_size as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            for i in 0..hidden_size {
                output_f32[offset + i] = row[i] * inv_rms * weight_f32[i];
            }
        }

        // Create output tensor
        let output =
            self.alloc_tensor(&node.output, input.1.clone(), DataType::F32)?;
        self.copy_to_tensor(
            &mut output.clone(),
            bytemuck::cast_slice(&output_f32),
        )?;
        graph.add_tensor(output);

        Ok(())
    }

    fn op_softmax(
        &self,
        _node: &OpNode,
        _inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        _graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // TODO: Use Metal softmax kernel
        Err(BackendError::OpNotSupported(
            "Metal softmax not yet implemented".to_string(),
        ))
    }

    fn op_silu(
        &self,
        _node: &OpNode,
        _inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        _graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // TODO: Use Metal SiLU kernel
        Err(BackendError::OpNotSupported(
            "Metal SiLU not yet implemented".to_string(),
        ))
    }

    fn op_rope(
        &self,
        _node: &OpNode,
        _inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        _graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // TODO: Use Metal RoPE kernel
        Err(BackendError::OpNotSupported(
            "Metal RoPE not yet implemented".to_string(),
        ))
    }

    fn op_matvec(
        &self,
        _node: &OpNode,
        _inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        _graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // TODO: Use Metal MV kernels
        Err(BackendError::OpNotSupported(
            "Metal MatVec not yet implemented".to_string(),
        ))
    }

    fn op_flash_attn(
        &self,
        _node: &OpNode,
        _inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        _graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // TODO: Use Metal Flash Attention kernel
        Err(BackendError::OpNotSupported(
            "Metal FlashAttn not yet implemented".to_string(),
        ))
    }

    fn op_add(
        &self,
        _node: &OpNode,
        _inputs: Vec<(String, Vec<usize>, DataType, std::rc::Rc<Buffer>)>,
        _graph: &mut ComputeGraph,
    ) -> BackendResult<()> {
        // TODO: Use Metal Add kernel
        Err(BackendError::OpNotSupported(
            "Metal Add not yet implemented".to_string(),
        ))
    }
}
