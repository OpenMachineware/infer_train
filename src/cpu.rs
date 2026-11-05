// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// CPU execution module

pub mod flash_attention;
pub mod graph_executor;
pub mod thread_pool;
pub mod tiled_gemv;
pub mod work_buffers;

pub use flash_attention::{
    FlashAttnBatchScratch, FlashAttnPrefillScratch, FlashAttnScratch,
    flash_attention_prefill_optimized, flash_attention_single,
};
pub use graph_executor::{GraphWorkBuffers, execute_graph_parallel};
pub use thread_pool::parallel_work_stealing as gemv_work_stealing;
pub use thread_pool::{
    GEMVOp, SpinBarrier, SyncPtr, ThreadPool, ThreadPoolState,
    default_n_threads, execute_layer_with_barrier, gemv_batch_barrier,
    global_pool, parallel_chunks, parallel_work_stealing,
};
pub use tiled_gemv::{
    gemv_q4k_q8k_tiled, gemv_q5k_q8k_tiled, gemv_q6k_q8k_tiled,
};
pub use work_buffers::{CpuWorkBuffers, CpuWorkBuffersPool};
