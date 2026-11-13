// Quantized KV cache (PagedAttention)

use crate::ops::QuantType;

/// Block size
pub const BLOCK_SIZE: usize = 16;

/// Quantized KV cache block
#[allow(dead_code)]
pub struct QuantizedKVBlock {
    key: Vec<u8>,
    value: Vec<u8>,
    seq_len: u16,
    layer_idx: u8,
}

/// Block pool
pub struct BlockPool {
    free_list: Vec<usize>,
    ref_counts: Vec<u32>,
    quant_type: QuantType,
}

impl BlockPool {
    pub fn new(capacity: usize, quant_type: QuantType) -> Self {
        Self {
            free_list: (0..capacity).collect(),
            ref_counts: vec![0; capacity],
            quant_type,
        }
    }

    pub fn allocate(&mut self) -> Option<usize> {
        if let Some(id) = self.free_list.pop() {
            self.ref_counts[id] = 1;
            return Some(id);
        }
        None
    }

    pub fn share(&mut self, id: usize) {
        self.ref_counts[id] += 1;
    }

    pub fn free(&mut self, id: usize) {
        self.ref_counts[id] -= 1;
        if self.ref_counts[id] == 0 {
            self.free_list.push(id);
        }
    }

    pub fn quant_type(&self) -> QuantType {
        self.quant_type
    }
}
