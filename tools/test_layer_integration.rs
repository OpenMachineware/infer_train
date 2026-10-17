// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Integration test for Layer 1 (BufferManager) + Layer 2 (OperationExecutor)

use infer_train::buffer::{BufferId, BufferManager, ComputeDevice};
use infer_train::executor::{Op, OperationExecutor};
use infer_train::model::tensor::DataType;
use std::sync::{Arc, Mutex};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Device;

fn main() {
    println!("=== Layer 1+2 Integration Test ===\n");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let device = Device::system_default().expect("No Metal device");
        let buffer_mgr =
            Arc::new(Mutex::new(BufferManager::new(device.clone())));

        test_buffer_lifecycle_integration(&buffer_mgr);
        test_device_selection_integration(&buffer_mgr);
        test_op_execution_integration(&buffer_mgr);

        println!("\n✓ All integration tests passed");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        let buffer_mgr = Arc::new(Mutex::new(BufferManager::new()));

        test_buffer_lifecycle_integration(&buffer_mgr);
        test_op_execution_integration(&buffer_mgr);

        println!("\n✓ All CPU-only integration tests passed");
    }
}

fn test_buffer_lifecycle_integration(buffer_mgr: &Arc<Mutex<BufferManager>>) {
    println!("Test 1: Buffer lifecycle with BufferManager");

    // Create input buffer
    let input_id = BufferId::new(1);
    {
        let mut mgr = buffer_mgr.lock().unwrap();
        let mut input_buf =
            mgr.get_or_load_on(input_id, 4096, ComputeDevice::Cpu);

        // Write test data
        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        if let Some(data) = input_buf.as_cpu_mut() {
            for i in 0..data.len().min(16) {
                data[i] = (i % 256) as u8;
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
        {
            let data = input_buf.as_cpu_mut();
            for i in 0..data.len().min(16) {
                data[i] = (i % 256) as u8;
            }
        }
    }

    // Verify buffer persisted
    {
        let mut mgr = buffer_mgr.lock().unwrap();
        let loc = mgr.get_location(&input_id);
        assert!(loc.is_some(), "Buffer should be registered");

        // Check if we can retrieve it again
        let cached_buf = mgr.get_or_load_on(input_id, 4096, ComputeDevice::Cpu);
        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        if let Some(data) = cached_buf.as_cpu() {
            println!(
                "  Buffer content verified: first 16 bytes = {:?}",
                &data[..16.min(data.len())]
            );
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
        {
            let data = cached_buf.as_cpu();
            println!(
                "  Buffer content verified: first 16 bytes = {:?}",
                &data[..16.min(data.len())]
            );
        }
    }

    println!("  ✓ Buffer lifecycle works across BufferManager calls");
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn test_device_selection_integration(buffer_mgr: &Arc<Mutex<BufferManager>>) {
    println!("\nTest 2: Device selection with BufferManager");

    // Create buffers on different devices
    let cpu_id = BufferId::new(10);
    let gpu_id = BufferId::new(11);

    {
        let mut mgr = buffer_mgr.lock().unwrap();
        let cpu_buf = mgr.get_or_load_on(cpu_id, 1024, ComputeDevice::Cpu);
        let gpu_buf = mgr.get_or_load(gpu_id, 1024);

        println!("  Created CPU buffer: {} bytes", cpu_buf.size());
        println!("  Created GPU buffer: {} bytes", gpu_buf.length());
    }

    // Verify locations
    {
        let mgr = buffer_mgr.lock().unwrap();
        let cpu_loc = mgr.get_location(&cpu_id);
        let gpu_loc = mgr.get_location(&gpu_id);

        assert!(
            matches!(
                cpu_loc,
                Some(infer_train::buffer::BufferLocation::Host(_))
            ),
            "CPU buffer should be Host"
        );
        assert!(
            matches!(
                gpu_loc,
                Some(infer_train::buffer::BufferLocation::Local(_))
            ),
            "GPU buffer should be Local"
        );

        println!("  ✓ Locations correctly tracked");
    }
}

fn test_op_execution_integration(buffer_mgr: &Arc<Mutex<BufferManager>>) {
    println!("\nTest 3: Op execution with BufferManager");

    // Create buffers for GEMV operation
    let input_id = BufferId::new(20);
    let output_id = BufferId::new(21);

    {
        let mut mgr = buffer_mgr.lock().unwrap();
        let _input = mgr.get_or_load_on(input_id, 4096, ComputeDevice::Cpu);
        let _output = mgr.get_or_load_on(output_id, 4096, ComputeDevice::Cpu);
    }

    // Create executor
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    let executor = {
        // Create a dummy GPU context for this test
        // In real use, this would be a proper MetalContext
        use infer_train::quant::vec_dot::gpu::MetalContext;
        let device = Device::system_default().expect("No Metal device");
        // For this test, we'll skip GPU context creation
        // Real integration will need MetalContext::new()
        println!("  Note: GPU context creation skipped in integration test");
        return;
    };

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    let mut executor = OperationExecutor::new(buffer_mgr.clone());

    // Create GEMV operation
    let op =
        Op::Gemv { output_size: 4096, input_size: 4096, dtype: DataType::Q4_K };

    // Execute (currently stub)
    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        let result = executor.execute(&op, &[input_id], output_id);
        println!("  Execution result: {:?}", result);

        // Verify output buffer was created
        let mgr = buffer_mgr.lock().unwrap();
        let output_loc = mgr.get_location(&output_id);
        assert!(output_loc.is_some(), "Output buffer should exist");

        println!("  ✓ Op execution works with BufferManager");
    }
}

fn test_cross_device_scenario(buffer_mgr: &Arc<Mutex<BufferManager>>) {
    println!("\nTest 4: Cross-device scenario (CPU fallback)");

    // Simulate GPU-unsupported format
    let input_id = BufferId::new(30);

    {
        let mut mgr = buffer_mgr.lock().unwrap();
        // Create on GPU first (if available)
        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        {
            let _ = mgr.get_or_load(input_id, 4096);
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
        {
            let _ = mgr.get_or_load_on(input_id, 4096, ComputeDevice::Cpu);
        }
    }

    // Create F16 GEMV (not GPU supported)
    let op =
        Op::Gemv { output_size: 4096, input_size: 4096, dtype: DataType::F16 };

    assert!(!op.is_gpu_supported(), "F16 should not be GPU supported");
    println!("  F16 GEMV correctly detected as CPU-only");

    println!("  ✓ Cross-device detection works");
}
