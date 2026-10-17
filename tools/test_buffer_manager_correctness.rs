// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Comprehensive correctness test for BufferManager (Layer 1)

use infer_train::buffer::{
    BufferId, BufferLocation, BufferManager, BufferRef, ComputeDevice,
};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Device;

fn main() {
    println!("=== BufferManager Correctness Test ===\n");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let device = Device::system_default().expect("No Metal device");
        let mut manager = BufferManager::new(device);

        test_gpu_buffer_lifecycle(&mut manager);
        test_cpu_buffer_lifecycle(&mut manager);
        test_location_tracking(&mut manager);
        test_pin_unpin(&mut manager);
        test_eviction(&mut manager);
        test_mixed_devices(&mut manager);
        test_pool_reuse(&mut manager);

        println!("\n✓ All correctness tests passed");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        let mut manager = BufferManager::new();

        test_cpu_buffer_lifecycle(&mut manager);
        test_location_tracking(&mut manager);
        test_pin_unpin(&mut manager);
        test_eviction(&mut manager);
        test_pool_reuse(&mut manager);

        println!("\n✓ All CPU-only tests passed");
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn test_gpu_buffer_lifecycle(manager: &mut BufferManager) {
    println!("Test 1: GPU buffer lifecycle");

    // Create buffer
    let id1 = BufferId::new(1001);
    let buf1 = manager.get_or_load(id1, 1024);
    assert_eq!(buf1.length(), 1024, "Buffer size should match requested size");

    // Retrieve same buffer
    let buf1_again = manager.get_or_load(id1, 1024);
    assert_eq!(
        buf1.length(),
        buf1_again.length(),
        "Same buffer should be returned for same ID"
    );

    // Check stats
    let stats = manager.stats();
    assert_eq!(stats.cached_gpu, 1, "Should have 1 GPU buffer cached");
    assert_eq!(
        stats.pool_gpu.total_created, 1,
        "Should have created 1 GPU buffer"
    );

    println!("  ✓ GPU buffer create, retrieve, stats");
}

fn test_cpu_buffer_lifecycle(manager: &mut BufferManager) {
    println!("\nTest 2: CPU buffer lifecycle");

    // Create buffer via get_or_load_on
    let id1 = BufferId::new(2001);
    let buf1 = manager.get_or_load_on(id1, 2048, ComputeDevice::Cpu);
    match &buf1 {
        BufferRef::Cpu(data) => {
            assert_eq!(data.len(), 2048, "CPU buffer size should match");
        }
        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        BufferRef::Gpu(_) => panic!("Expected CPU buffer, got GPU"),
    }

    // Retrieve same buffer
    let buf1_again = manager.get_or_load_on(id1, 2048, ComputeDevice::Cpu);
    assert!(
        matches!((&buf1, &buf1_again), (BufferRef::Cpu(a), BufferRef::Cpu(b)) if a.len() == b.len()),
        "Same CPU buffer should be returned"
    );

    // Check stats
    let stats = manager.stats();
    assert_eq!(stats.cached_cpu, 1, "Should have 1 CPU buffer cached");
    assert_eq!(
        stats.pool_cpu.total_created, 1,
        "Should have created 1 CPU buffer"
    );

    println!("  ✓ CPU buffer create, retrieve, stats");
}

fn test_location_tracking(manager: &mut BufferManager) {
    println!("\nTest 3: Location tracking");

    // GPU buffer (if available)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let id_gpu = BufferId::new(3001);
        let _ = manager.get_or_load(id_gpu, 4096);

        let loc = manager.get_location(&id_gpu);
        assert!(loc.is_some(), "GPU buffer should have location");
        assert!(
            matches!(loc.unwrap(), BufferLocation::Local(_)),
            "Should be Local GPU"
        );
    }

    // CPU buffer
    let id_cpu = BufferId::new(3002);
    let _ = manager.get_or_load_on(id_cpu, 4096, ComputeDevice::Cpu);

    let loc = manager.get_location(&id_cpu);
    assert!(loc.is_some(), "CPU buffer should have location");
    assert!(matches!(loc.unwrap(), BufferLocation::Host(_)), "Should be Host");

    // Unknown buffer
    let id_unknown = BufferId::new(9999);
    let loc = manager.get_location(&id_unknown);
    assert!(loc.is_none(), "Unknown buffer should have no location");

    println!("  ✓ Location tracking for GPU, CPU, unknown");
}

fn test_pin_unpin(manager: &mut BufferManager) {
    println!("\nTest 4: Pin/Unpin mechanism");

    let id1 = BufferId::new(4001);
    let _ = manager.get_or_load_on(id1, 1024, ComputeDevice::Cpu);

    // Pin
    manager.pin(id1);
    assert!(manager.is_pinned(&id1), "Buffer should be pinned");

    // Unpin
    manager.unpin(id1);
    assert!(!manager.is_pinned(&id1), "Buffer should be unpinned");

    // Unknown buffer
    let id_unknown = BufferId::new(9999);
    assert!(
        !manager.is_pinned(&id_unknown),
        "Unknown buffer should not be pinned"
    );

    println!("  ✓ Pin/Unpin works correctly");
}

fn test_eviction(manager: &mut BufferManager) {
    println!("\nTest 5: Eviction");

    // Clear existing state first
    manager.evict();

    // Create multiple buffers
    let id1 = BufferId::new(5001);
    let id2 = BufferId::new(5002);
    let id3 = BufferId::new(5003);

    let _ = manager.get_or_load_on(id1, 1024, ComputeDevice::Cpu);
    let _ = manager.get_or_load_on(id2, 2048, ComputeDevice::Cpu);
    let _ = manager.get_or_load_on(id3, 4096, ComputeDevice::Cpu);

    let stats = manager.stats();
    assert_eq!(stats.cached_cpu, 3, "Should have 3 CPU buffers cached");

    // Pin one buffer
    manager.pin(id2);

    // Evict unpinned buffers
    manager.evict();

    let stats = manager.stats();
    assert_eq!(stats.cached_cpu, 1, "Should have 1 CPU buffer left (pinned)");
    assert_eq!(stats.pinned, 1, "Should have 1 pinned buffer");

    // Pinned buffer should still be accessible
    let loc = manager.get_location(&id2);
    assert!(loc.is_some(), "Pinned buffer should still have location");

    // Evicted buffers should have new location
    let loc = manager.get_location(&id1);
    assert!(
        matches!(loc, Some(BufferLocation::Remote { .. })),
        "Evicted buffer should be Remote"
    );

    println!("  ✓ Eviction respects pinning");
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn test_mixed_devices(manager: &mut BufferManager) {
    println!("\nTest 6: Mixed GPU/CPU buffers");

    // Create GPU buffer
    let id_gpu = BufferId::new(6001);
    let _ = manager.get_or_load(id_gpu, 1024);

    // Create CPU buffer
    let id_cpu = BufferId::new(6002);
    let _ = manager.get_or_load_on(id_cpu, 2048, ComputeDevice::Cpu);

    // Create another GPU buffer
    let id_gpu2 = BufferId::new(6003);
    let _ = manager.get_or_load(id_gpu2, 4096);

    // Check stats
    let stats = manager.stats();
    assert_eq!(stats.cached_gpu, 2, "Should have 2 GPU buffers");
    assert_eq!(stats.cached_cpu, 1, "Should have 1 CPU buffer");

    // Pin one of each
    manager.pin(id_gpu);
    manager.pin(id_cpu);

    // Evict
    manager.evict();

    let stats = manager.stats();
    assert_eq!(stats.cached_gpu, 1, "Should have 1 GPU buffer left");
    assert_eq!(stats.cached_cpu, 1, "Should have 1 CPU buffer left");

    println!("  ✓ Mixed GPU/CPU management works");
}

fn test_pool_reuse(manager: &mut BufferManager) {
    println!("\nTest 7: Pool reuse efficiency");

    // Create and release multiple same-size buffers
    let sizes = [1024, 2048, 4096];
    let mut ids = Vec::new();

    for (i, &size) in sizes.iter().enumerate() {
        let id = BufferId::new(7000 + i as u64);
        let _ = manager.get_or_load_on(id, size, ComputeDevice::Cpu);
        ids.push(id);
    }

    let stats_before = manager.stats();
    println!(
        "  Before: created {} buffers",
        stats_before.pool_cpu.total_created
    );

    // Evict all
    manager.evict();

    // Create same-size buffers again
    for (i, &size) in sizes.iter().enumerate() {
        let id = BufferId::new(8000 + i as u64);
        let _ = manager.get_or_load_on(id, size, ComputeDevice::Cpu);
    }

    let stats_after = manager.stats();
    println!("  After: created {} buffers", stats_after.pool_cpu.total_created);

    // Should reuse from pool, not create new
    assert_eq!(
        stats_before.pool_cpu.total_created, stats_after.pool_cpu.total_created,
        "Pool should reuse buffers, not create new ones"
    );

    println!("  ✓ Pool reuse works correctly");
}
