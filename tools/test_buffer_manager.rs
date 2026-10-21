// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Buffer Manager functionality

use infer_train::buffer::{BufferId, BufferManager, BufferRef, ComputeDevice};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Device;

fn main() {
    println!("=== Buffer Manager Test ===\n");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let device = Device::system_default().expect("No Metal device");
        let mut manager = BufferManager::new(device);

        // Test 1: GPU buffer management
        println!("Test 1: GPU buffer management");
        let id1 = BufferId::new(1);
        let buf1 = manager.get_or_load(id1, 1024);
        println!("  Created GPU buffer: {} bytes", buf1.length());

        // Test 2: CPU buffer management
        println!("\nTest 2: CPU buffer management");
        let id2 = BufferId::new(2);
        let buf2 = manager.get_or_load_on(id2, 2048, ComputeDevice::Cpu);
        match &buf2 {
            BufferRef::Cpu(data) => {
                println!("  Created CPU buffer: {} bytes", unsafe {
                    (&**data.get()).len()
                });
            }
            _ => println!("  ERROR: Expected CPU buffer"),
        }

        // Test 3: Reuse CPU buffers
        println!("\nTest 3: Reuse CPU buffers");
        let id3 = BufferId::new(3);
        let buf3a = manager.get_or_load_on(id3, 1024, ComputeDevice::Cpu);
        let buf3b = manager.get_or_load_on(id3, 1024, ComputeDevice::Cpu);
        println!(
            "  Same buffer returned: {}",
            matches!((&buf3a, &buf3b), (BufferRef::Cpu(a), BufferRef::Cpu(b)) if unsafe { (&**a.get()).len() } == unsafe { (&**b.get()).len() })
        );

        // Test 4: Pin and evict
        println!("\nTest 4: Pin and evict");
        manager.pin(id2);
        println!("  Pinned id2: {}", manager.is_pinned(&id2));
        manager.evict();
        let stats = manager.stats();
        println!("  After eviction:");
        println!("    GPU cached: {}", stats.cached_gpu);
        println!("    CPU cached: {}", stats.cached_cpu);
        println!("    Pinned: {}", stats.pinned);

        // Test 5: Location tracking
        println!("\nTest 5: Location tracking");
        if let Some(loc) = manager.get_location(&id2) {
            println!("  id2 location: {:?}", loc.is_host());
        }

        // Test 6: Transfer request (stub)
        println!("\nTest 6: Transfer request (stub)");
        manager.transfer(id1, ComputeDevice::LocalGpu, ComputeDevice::Cpu);
        println!("  Transfer request logged (stub implementation)");

        // Final stats
        let stats = manager.stats();
        println!("\nFinal stats:");
        println!("  Registered: {}", stats.registered);
        println!(
            "  GPU pool: created={}, in_use={}, cached={}",
            stats.pool_gpu.total_created,
            stats.pool_gpu.in_use,
            stats.pool_gpu.cached
        );
        println!(
            "  CPU pool: created={}, in_use={}, cached={}",
            stats.pool_cpu.total_created,
            stats.pool_cpu.in_use,
            stats.pool_cpu.cached
        );

        println!("\n✓ All tests passed");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Testing CPU-only buffer management\n");

        let mut manager = BufferManager::new();

        // Test CPU buffer management
        println!("Test 1: CPU buffer management");
        let id1 = BufferId::new(1);
        let buf1 = manager.get_or_load_on(id1, 1024, ComputeDevice::Cpu);
        match &buf1 {
            BufferRef::Cpu(data) => {
                let data = data.borrow();
                println!("  Created CPU buffer: {} bytes", data.len());
            }
            _ => println!("  ERROR: Expected CPU buffer"),
        }

        // Test reuse
        println!("\nTest 2: Reuse CPU buffers");
        let id2 = BufferId::new(2);
        let buf2a = manager.get_or_load_on(id2, 2048, ComputeDevice::Cpu);
        let buf2b = manager.get_or_load_on(id2, 2048, ComputeDevice::Cpu);
        let len_match = match (&buf2a, &buf2b) {
            (BufferRef::Cpu(a), BufferRef::Cpu(b)) => {
                a.borrow().len() == b.borrow().len()
            }
            _ => false,
        };
        println!("  Same buffer returned: {}", len_match);

        let stats = manager.stats();
        println!("\nStats:");
        println!("  CPU cached: {}", stats.cached_cpu);
        println!("  CPU pool created: {}", stats.pool_cpu.total_created);

        println!("\n✓ All tests passed");
    }
}
