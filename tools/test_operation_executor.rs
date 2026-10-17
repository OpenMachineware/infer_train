// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test OperationExecutor (Layer 2) basic functionality

use infer_train::buffer::{BufferId, BufferManager, ComputeDevice};
use infer_train::executor::{Op, OperationExecutor};
use std::sync::{Arc, Mutex};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Device;

fn main() {
    println!("=== OperationExecutor Test ===\n");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let device = Device::system_default().expect("No Metal device");
        let buffer_mgr =
            Arc::new(Mutex::new(BufferManager::new(device.clone())));
        let gpu_ctx = Arc::new(Mutex::new(create_gpu_context(&device)));
        let executor = OperationExecutor::new(buffer_mgr.clone(), gpu_ctx);

        test_device_selection(&executor);
        test_op_types();
        test_flop_estimates();

        println!("\n✓ All tests passed");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        let buffer_mgr = Arc::new(Mutex::new(BufferManager::new()));
        let executor = OperationExecutor::new(buffer_mgr);

        test_op_types();
        test_flop_estimates();

        println!("\n✓ All CPU-only tests passed");
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn create_gpu_context(
    device: &Device,
) -> infer_train::quant::vec_dot::gpu::MetalContext {
    use infer_train::quant::vec_dot::gpu::MetalContext;
    // Create a minimal GPU context for testing
    // In real use, this would be created by MetalContext::new()
    // For now, we'll skip actual GPU context creation in this test
    unimplemented!("GPU context creation not yet integrated")
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn test_device_selection(executor: &OperationExecutor) {
    println!("Test 1: Device selection");

    // GPU-first by default
    assert_eq!(executor.preferred_device(), ComputeDevice::LocalGpu);

    println!("  ✓ Default device is LocalGpu");
}

fn test_op_types() {
    println!("\nTest 2: Op types");

    // Test GEMV
    let op_gemv = Op::Gemv {
        output_size: 4096,
        input_size: 4096,
        dtype: infer_train::model::tensor::DataType::Q4_K,
    };
    assert!(op_gemv.is_gpu_supported(), "Q4_K GEMV should be GPU supported");

    // Test unsupported format
    let op_f16 = Op::Gemv {
        output_size: 4096,
        input_size: 4096,
        dtype: infer_train::model::tensor::DataType::F16,
    };
    assert!(!op_f16.is_gpu_supported(), "F16 GEMV should use CPU fallback");

    // Test other ops
    let op_rms = Op::RmsNorm { size: 4096, eps: 1e-5 };
    assert!(op_rms.is_gpu_supported(), "RMS Norm should be GPU supported");

    let op_flash = Op::FlashAttention {
        seq_len: 128,
        n_heads: 32,
        n_kv_heads: 32,
        head_dim: 128,
    };
    assert!(
        op_flash.is_gpu_supported(),
        "Flash Attention should be GPU supported"
    );

    println!("  ✓ Op type detection works");
}

fn test_flop_estimates() {
    println!("\nTest 3: FLOP estimates");

    let op_gemv = Op::Gemv {
        output_size: 4096,
        input_size: 4096,
        dtype: infer_train::model::tensor::DataType::Q4_K,
    };
    let flops = op_gemv.estimated_flops();
    println!("  GEMV 4096x4096: {} FLOPs", flops);
    assert_eq!(flops, 2 * 4096 * 4096, "GEMV FLOPs should be 2 * M * N");

    let op_flash = Op::FlashAttention {
        seq_len: 128,
        n_heads: 32,
        n_kv_heads: 32,
        head_dim: 128,
    };
    let flops = op_flash.estimated_flops();
    println!("  Flash Attention 128x32x128: {} FLOPs", flops);

    println!("  ✓ FLOP estimates work");
}
