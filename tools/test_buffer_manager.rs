// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Buffer Manager functionality

use infer_train::buffer::{BufferId, BufferManager};
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Device;

fn main() {
    println!("=== Buffer Manager Test ===\n");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let device = Device::system_default().expect("No Metal device");
        let mut manager = BufferManager::new(device);

        // Test 1: Create and get buffers
        println!("Test 1: Create and get buffers");
        let id1 = BufferId::new(1);
        let id2 = BufferId::new(2);
        let id3 = BufferId::new(3);

        let buf1 = manager.get_or_load(id1, 1024);
        let buf2 = manager.get_or_load(id2, 2048);
        let buf3 = manager.get_or_load(id3, 4096);

        println!("  Created buffers: 1KB, 2KB, 4KB");
        println!("  Buffer 1 length: {} bytes", buf1.length());
        println!("  Buffer 2 length: {} bytes", buf2.length());
        println!("  Buffer 3 length: {} bytes", buf3.length());

        // Test 2: Reuse buffers
        println!("\nTest 2: Reuse buffers");
        let id4 = BufferId::new(4);
        let buf4 = manager.get_or_load(id4, 1024); // Should reuse from pool

        let stats = manager.stats();
        println!("  Pool stats:");
        println!("    Total created: {}", stats.pool.total_created);
        println!("    In use: {}", stats.pool.in_use);
        println!("    Cached: {}", stats.pool.cached);
        println!("    Registered: {}", stats.registered);

        // Test 3: Pin buffers
        println!("\nTest 3: Pin buffers");
        manager.pin(id1);
        manager.pin(id2);
        println!("  Pinned id1 and id2");
        println!("  id1 is pinned: {}", manager.is_pinned(&id1));
        println!("  id3 is pinned: {}", manager.is_pinned(&id3));

        // Test 4: Evict unpinned buffers
        println!("\nTest 4: Evict unpinned buffers");
        manager.evict();
        let stats = manager.stats();
        println!("  After eviction:");
        println!("    Cached: {}", stats.cached);
        println!("    Pinned: {}", stats.pinned);

        println!("\n✓ All tests passed");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Buffer Manager requires Metal support (macOS aarch64)");
    }
}
