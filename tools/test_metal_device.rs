// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: Verify MetalDevice singleton and basic operations

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::quant::vec_dot::gpu::device::MetalDevice;

        // Get the global device (singleton)
        let device1 = MetalDevice::global();
        let device2 = MetalDevice::global();

        // Verify singleton - same pointer
        assert!(std::ptr::eq(device1, device2));
        println!("✓ MetalDevice is a singleton");

        // Check device name
        let device = device1.device();
        println!("✓ Device: {}", device.name());

        // Create a test buffer
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let buffer = device1.new_buffer_with_data(bytemuck::cast_slice(&data));
        println!("✓ Created buffer of size {} bytes", buffer.length());

        // Create a command buffer and verify queue works
        let queue = device1.queue();
        let cb = queue.new_command_buffer();
        println!("✓ Created command buffer from shared queue");

        // Commit and wait
        cb.commit();
        cb.wait_until_completed();
        println!("✓ Command buffer executed successfully");

        println!("\n=== MetalDevice Test Passed ===");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
