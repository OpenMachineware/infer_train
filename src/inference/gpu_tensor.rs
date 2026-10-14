// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GPU tensor abstraction for zero-copy operations

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

/// GPU-resident tensor (zero-copy between operations)
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct GpuTensor {
    pub buffer: Rc<Buffer>,
    pub shape: Vec<usize>,
    pub dtype: crate::model::tensor::DataType,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl GpuTensor {
    pub fn new(
        buffer: Rc<Buffer>,
        shape: Vec<usize>,
        dtype: crate::model::tensor::DataType,
    ) -> Self {
        Self { buffer, shape, dtype }
    }

    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Read back to CPU (only at final output)
    pub fn to_vec_f32(&self) -> Vec<f32> {
        let bytes = self.buffer.contents();
        let byte_len = self.buffer.length() as usize;
        let n = self.elem_count();
        let mut result = vec![0.0f32; n];

        // For now, assume F32 dtype
        // TODO: Handle quantized formats
        unsafe {
            let byte_slice =
                std::slice::from_raw_parts(bytes as *const u8, byte_len);
            for i in 0..n {
                let offset = i * 4;
                if offset + 4 <= byte_len {
                    let b: [u8; 4] =
                        byte_slice[offset..offset + 4].try_into().unwrap();
                    result[i] = f32::from_le_bytes(b);
                }
            }
        }

        result
    }
}
