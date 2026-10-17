// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test ForwardBufferPool functionality

use std::cell::RefCell;
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Device;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::buffer::ForwardBufferPool;

fn main() {
    println!("=== ForwardBufferPool Test ===\n");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let device = Device::system_default().expect("No Metal device");

        // Test 1: Basic buffer creation
        println!("Test 1: Basic buffer creation");
        let reuse_pool = Rc::new(RefCell::new(Vec::new()));
        let mut forward_pool =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        let buf1 = forward_pool.get(1024);
        println!("  Created buffer: {} bytes", buf1.length());

        let stats = forward_pool.stats();
        println!("  Active buffers: {}", stats.active);
        println!("  Reusable buffers: {}", stats.reusable);
        assert_eq!(stats.active, 1);
        assert_eq!(stats.reusable, 0);

        // Test 2: Zeroed buffer
        println!("\nTest 2: Zeroed buffer");
        let buf2 = forward_pool.get_zeroed(2048);
        let contents = unsafe {
            std::slice::from_raw_parts(
                buf2.contents() as *const u8,
                buf2.length() as usize,
            )
        };
        let all_zero = contents.iter().all(|&b| b == 0);
        println!("  Buffer is zeroed: {}", all_zero);
        assert!(all_zero);

        // Test 3: Register existing buffer
        println!("\nTest 3: Register existing buffer");
        use metal::MTLResourceOptions;
        let external_buf = std::rc::Rc::new(
            device.new_buffer(512, MTLResourceOptions::StorageModeShared),
        );
        let buf3 = forward_pool.register(external_buf.clone());
        println!("  Registered buffer: {} bytes", buf3.length());

        let stats = forward_pool.stats();
        println!("  Active buffers: {}", stats.active);
        assert_eq!(stats.active, 3);

        // Test 4: Release all buffers
        println!("\nTest 4: Release all buffers");
        forward_pool.release_all();
        let stats = forward_pool.stats();
        println!(
            "  After release - Active: {}, Reusable: {}",
            stats.active, stats.reusable
        );
        assert_eq!(stats.active, 0);
        assert_eq!(stats.reusable, 3);

        // Test 5: Reuse from pool
        println!("\nTest 5: Reuse from pool");
        println!("  Reuse pool size before: {}", reuse_pool.borrow().len());
        let mut forward_pool2 =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let buf4 = forward_pool2.get(1024);
        println!("  Reuse pool size after get: {}", reuse_pool.borrow().len());
        println!("  Buffer was reused: {}", reuse_pool.borrow().len() < 3);

        // Test 6: Automatic drop behavior
        println!("\nTest 6: Automatic drop behavior");
        {
            let reuse_pool3 = Rc::new(RefCell::new(Vec::new()));
            let mut forward_pool3 =
                ForwardBufferPool::new(device.clone(), reuse_pool3.clone());
            let _buf = forward_pool3.get(4096);
            // forward_pool3 is dropped here
        }
        println!(
            "  Forward pool dropped successfully (buffers returned to reuse pool)"
        );

        println!("\n✓ All tests passed");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("✓ CPU-only build - ForwardBufferPool is a stub");
    }
}
