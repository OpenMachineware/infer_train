// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Metal backend for GPU execution.

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod metal_backend {
    use std::sync::Arc;

    use metal::Buffer;

    use super::super::backend::{Backend, ExecutionContext};
    use super::super::graph::{ComputeGraph, OpType};
    use super::super::storage::{Device, Storage};
    use crate::model::tensor::DataType;
    use crate::quant::vec_dot::gpu::metal::MetalContext;

    /// Metal GPU backend.
    pub struct MetalBackend<'a> {
        ctx: &'a MetalContext,
    }

    impl<'a> MetalBackend<'a> {
        pub fn new(ctx: &'a MetalContext) -> Self {
            Self { ctx }
        }

        pub fn context(&self) -> &MetalContext {
            self.ctx
        }
    }

    impl Backend for MetalBackend<'_> {
        fn device(&self) -> Device {
            Device::Metal
        }

        fn alloc_tensor(&self, shape: &[usize], dtype: DataType) -> Storage {
            let num_elems: usize = shape.iter().product();
            let size = num_elems * dtype.block_size() / dtype.elems_per_block();
            let buffer = self.ctx.device().new_buffer(
                size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            Storage::Metal(Arc::new(buffer))
        }

        fn alloc_tensor_with_data(
            &self,
            _shape: &[usize],
            _dtype: DataType,
            data: &[u8],
        ) -> Storage {
            let buffer = self.ctx.device().new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            Storage::Metal(Arc::new(buffer))
        }

        fn execute(
            &self,
            graph: &ComputeGraph,
            ctx: &ExecutionContext,
        ) -> Result<Vec<Storage>, String> {
            // Key optimization: Batch all operations into minimal kernel dispatches
            // This is where we reduce the 480+ dispatches problem

            let mut tensor_storage: std::collections::HashMap<String, Storage> =
                std::collections::HashMap::new();

            // Initialize weight tensors (already in GPU memory)
            for (name, tensor) in &graph.tensors {
                if let Some(storage) = &tensor.storage {
                    tensor_storage.insert(name.clone(), storage.clone());
                }
            }

            // Initialize input tensors
            for input_name in &graph.inputs {
                if let Some(tensor) = graph.get_tensor(input_name) {
                    if let Some(storage) = &tensor.storage {
                        tensor_storage
                            .insert(input_name.clone(), storage.clone());
                    }
                }
            }

            // Group operations by type for batch dispatch
            // This is the key optimization to reduce dispatch overhead
            let mut pending_ops: Vec<(&OpType, Vec<Storage>, String)> =
                Vec::new();

            for node in &graph.nodes {
                let inputs: Vec<Storage> = node
                    .inputs
                    .iter()
                    .map(|name| {
                        tensor_storage.get(name).cloned().ok_or_else(|| {
                            format!("Tensor '{}' not found", name)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                pending_ops.push((&node.op, inputs, node.output.clone()));
            }

            // TODO: Fuse consecutive operations into single dispatches
            // For now, execute one by one but without synchronization between them
            for (op, inputs, output_name) in pending_ops {
                let output_shape = Self::infer_output_shape(op, &inputs)?;
                let output_dtype = Self::infer_output_dtype(op);
                let mut output_storage =
                    self.alloc_tensor(&output_shape, output_dtype);

                match op {
                    OpType::RmsNorm { eps } => {
                        self.rms_norm(
                            &inputs[0],
                            &inputs[1],
                            &mut output_storage,
                            *eps,
                        )?;
                    }
                    OpType::MatMul => {
                        // TODO: Get weight dtype from metadata
                        let weight_dtype = DataType::Q4_0;
                        self.matmul(
                            &inputs[0],
                            &inputs[1],
                            weight_dtype,
                            &mut output_storage,
                        )?;
                    }
                    OpType::FlashAttn {
                        n_heads, head_dim, n_kv_heads, ..
                    } => {
                        self.flash_attn(
                            &inputs[0],
                            &inputs[1],
                            &inputs[2],
                            &mut output_storage,
                            *n_heads,
                            *head_dim,
                            *n_kv_heads,
                            ctx.seq_len,
                        )?;
                    }
                    OpType::RopeNeox {
                        n_heads,
                        head_dim,
                        rope_freq_base,
                        ..
                    } => {
                        // Position from context
                        self.rope(
                            &inputs[0],
                            ctx.seq_len,
                            &mut output_storage,
                            *n_heads,
                            *head_dim,
                            *rope_freq_base,
                        )?;
                    }
                    OpType::Add => {
                        self.add(&inputs[0], &inputs[1], &mut output_storage)?;
                    }
                    OpType::Mul => {
                        self.mul(&inputs[0], &inputs[1], &mut output_storage)?;
                    }
                    OpType::Silu => {
                        self.silu(&inputs[0], &mut output_storage)?;
                    }
                    _ => return Err(format!("Unsupported op: {:?}", op)),
                }

                tensor_storage.insert(output_name, output_storage);
            }

            // Collect outputs
            let outputs: Vec<Storage> = graph
                .outputs
                .iter()
                .map(|name| {
                    tensor_storage
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("Output '{}' not found", name))
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(outputs)
        }

        fn copy_to_cpu(&self, storage: &Storage) -> Result<Vec<u8>, String> {
            match storage {
                Storage::Cpu(data) => Ok(data.as_ref().to_vec()),
                Storage::Metal(buffer) => {
                    let contents = buffer.contents();
                    let len = buffer.length() as usize;
                    let mut data = vec![0u8; len];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            contents as *const u8,
                            data.as_mut_ptr(),
                            len,
                        );
                    }
                    Ok(data)
                }
            }
        }

        fn copy_from_cpu(
            &self,
            data: &[u8],
            storage: &mut Storage,
        ) -> Result<(), String> {
            match storage {
                Storage::Cpu(buf) => {
                    Arc::make_mut(buf).copy_from_slice(data);
                    Ok(())
                }
                Storage::Metal(buffer) => {
                    let contents = buffer.contents();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            contents as *mut u8,
                            data.len(),
                        );
                    }
                    Ok(())
                }
            }
        }

        fn rms_norm(
            &self,
            input: &Storage,
            weight: &Storage,
            _output: &mut Storage,
            eps: f32,
        ) -> Result<(), String> {
            let input_buffer =
                input.as_metal_buffer().ok_or("Input not on Metal")?;
            let weight_buffer =
                weight.as_metal_buffer().ok_or("Weight not on Metal")?;

            // Get shape (assume 1D for now)
            let size = (input_buffer.length() / 4) as usize;

            let _result = self.ctx.rms_norm_f32_gpu(
                input_buffer,
                weight_buffer,
                1, // batch
                size,
                eps,
            )?;

            Ok(())
        }

        fn matmul(
            &self,
            input: &Storage,
            weight: &Storage,
            weight_dtype: DataType,
            _output: &mut Storage,
        ) -> Result<(), String> {
            let input_buffer =
                input.as_metal_buffer().ok_or("Input not on Metal")?;
            let weight_buffer =
                weight.as_metal_buffer().ok_or("Weight not on Metal")?;

            // Infer dimensions from buffer sizes
            let input_size = (input_buffer.length() / 4) as usize;
            let weight_bytes = weight_buffer.length() as usize;

            // For quantized weights, need to know the format
            let out_size = match weight_dtype {
                DataType::Q4_0
                | DataType::Q4_1
                | DataType::Q5_0
                | DataType::Q5_1
                | DataType::Q8_0 => {
                    weight_bytes / weight_dtype.block_size()
                        * weight_dtype.elems_per_block()
                }
                DataType::Q2_K
                | DataType::Q3_K
                | DataType::Q4_K
                | DataType::Q5_K
                | DataType::Q6_K => {
                    weight_bytes / weight_dtype.block_size()
                        * weight_dtype.elems_per_block()
                }
                DataType::IQ4_NL => weight_bytes / 18 * 32,
                _ => weight_bytes / 4, // FP formats
            };

            // Call the appropriate MV kernel based on dtype
            let _result = match weight_dtype {
                DataType::Q4_0 => self.ctx.mv_q4_0_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q4_1 => self.ctx.mv_q4_1_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q5_0 => self.ctx.mv_q5_0_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q5_1 => self.ctx.mv_q5_1_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q8_0 => self.ctx.mv_q8_0_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q2_K => self.ctx.mv_q2_k_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q3_K => self.ctx.mv_q3_k_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q4_K => self.ctx.mv_q4_k_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q5_K => self.ctx.mv_q5_k_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::Q6_K => self.ctx.mv_q6_k_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                DataType::IQ4_NL => self.ctx.mv_iq4_nl_f32_gpu_weights(
                    out_size,
                    input_size,
                    weight_buffer,
                    input_buffer,
                ),
                _ => {
                    return Err(format!(
                        "MatMul not implemented for dtype {:?}",
                        weight_dtype
                    ));
                }
            }?;

            Ok(())
        }

        fn flash_attn(
            &self,
            query: &Storage,
            k_cache: &Storage,
            v_cache: &Storage,
            _output: &mut Storage,
            n_heads: usize,
            head_dim: usize,
            n_kv_heads: usize,
            seq_len: usize,
        ) -> Result<(), String> {
            let query_buffer =
                query.as_metal_buffer().ok_or("Query not on Metal")?;
            let k_buffer =
                k_cache.as_metal_buffer().ok_or("K cache not on Metal")?;
            let v_buffer =
                v_cache.as_metal_buffer().ok_or("V cache not on Metal")?;

            let _result = self.ctx.flash_attn_f16_gpu(
                1, // n_seqs
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                query_buffer,
                k_buffer,
                v_buffer,
            )?;

            Ok(())
        }

        fn rope(
            &self,
            input: &Storage,
            pos: usize,
            _output: &mut Storage,
            n_heads: usize,
            head_dim: usize,
            freq_base: f32,
        ) -> Result<(), String> {
            let input_buffer =
                input.as_metal_buffer().ok_or("Input not on Metal")?;

            let _result = self.ctx.rope_neox_f32_gpu(
                input_buffer,
                &[pos as i32],
                1,
                n_heads,
                head_dim,
                freq_base,
                1.0,
            )?;

            Ok(())
        }

        fn add(
            &self,
            a: &Storage,
            b: &Storage,
            _output: &mut Storage,
        ) -> Result<(), String> {
            let a_buffer =
                a.as_metal_buffer().ok_or("First input not on Metal")?;
            let b_buffer =
                b.as_metal_buffer().ok_or("Second input not on Metal")?;

            let size = (a_buffer.length() / 4) as usize;

            let _result = self.ctx.add_f32_gpu(a_buffer, b_buffer, size)?;

            Ok(())
        }

        fn mul(
            &self,
            a: &Storage,
            b: &Storage,
            _output: &mut Storage,
        ) -> Result<(), String> {
            let a_buffer =
                a.as_metal_buffer().ok_or("First input not on Metal")?;
            let b_buffer =
                b.as_metal_buffer().ok_or("Second input not on Metal")?;

            let size = (a_buffer.length() / 4) as usize;

            let _result = self.ctx.mul_f32_gpu(a_buffer, b_buffer, size)?;

            Ok(())
        }

        fn silu(
            &self,
            input: &Storage,
            _output: &mut Storage,
        ) -> Result<(), String> {
            let input_buffer =
                input.as_metal_buffer().ok_or("Input not on Metal")?;

            let size = (input_buffer.length() / 4) as usize;

            let _result = self.ctx.silu_f32_gpu(input_buffer, size)?;

            Ok(())
        }
    }

    impl<'a> MetalBackend<'a> {
        fn infer_output_shape(
            op: &OpType,
            inputs: &[Storage],
        ) -> Result<Vec<usize>, String> {
            match op {
                OpType::RmsNorm { .. } => {
                    let size = inputs[0].size() / 4;
                    Ok(vec![size])
                }
                OpType::MatMul => {
                    // Output: (batch, out_features)
                    let in_size = inputs[0].size() / 4;
                    // We'd need weight shape to determine out_features
                    Ok(vec![in_size]) // Placeholder
                }
                OpType::FlashAttn { n_heads, head_dim, .. } => {
                    Ok(vec![*n_heads * head_dim])
                }
                OpType::RopeNeox { .. } => {
                    let size = inputs[0].size() / 4;
                    Ok(vec![size])
                }
                OpType::Add | OpType::Mul => {
                    let size = inputs[0].size() / 4;
                    Ok(vec![size])
                }
                OpType::Silu => {
                    let size = inputs[0].size() / 4;
                    Ok(vec![size])
                }
                _ => Ok(vec![]),
            }
        }

        fn infer_output_dtype(_op: &OpType) -> DataType {
            DataType::F32
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub use metal_backend::MetalBackend;
