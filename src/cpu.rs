// SPDX-License-Identifier: AGPL-3.0-or-later
// CPU execution module

pub mod graph_executor;
pub mod thread_pool;
pub mod tiled_gemv;
pub mod work_buffers;

pub use graph_executor::{GraphWorkBuffers, execute_graph_parallel};
pub use thread_pool::{
    GEMVOp, PersistentThreadPool, SpinBarrier, SyncPtr, default_n_threads,
    execute_layer_with_barrier, gemv_batch_barrier, gemv_work_stealing,
    global_pool, parallel_chunks, parallel_work_stealing,
};
pub use tiled_gemv::{
    gemv_q4k_q8k_tiled, gemv_q5k_q8k_tiled, gemv_q6k_q8k_tiled,
};
pub use work_buffers::{CpuWorkBuffers, CpuWorkBuffersPool};
