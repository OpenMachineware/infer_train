// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// CPU execution module

pub mod adaptive_scheduler;
pub mod flash_attention;
pub mod simd_gemm;
pub mod thread_pool;
pub mod tiled_gemv;
pub mod work_buffers;

pub use adaptive_scheduler::{AdaptiveScheduler, MatmulPath, get_scheduler};
pub use flash_attention::{
    FlashAttnBatchScratch, FlashAttnPrefillScratch, FlashAttnScratch,
    flash_attention_prefill_tiled, flash_attention_single,
};
pub use thread_pool::{
    SpinBarrier, SyncPtr, ThreadConfig, ThreadPool, ThreadPoolState,
    default_n_threads, global_pool, parallel_chunks, parallel_chunks_sized,
    parallel_work_stealing,
};
pub use tiled_gemv::{
    gemv_q4k_q8k_tiled, gemv_q5k_q8k_tiled, gemv_q6k_q8k_tiled,
};
pub use work_buffers::{CpuWorkBuffers, CpuWorkBuffersPool};
