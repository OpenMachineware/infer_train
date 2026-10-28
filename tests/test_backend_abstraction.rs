// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test backend abstraction.

#[cfg(test)]
mod tests {
    use infer_train::backend::{Device, DeviceType, MemoryTopology};

    #[test]
    fn test_device_detection() {
        let device = Device::detect_best();
        println!("Device: {}", device.name());
        println!("Type: {:?}", device.device_type());
        println!("Memory: {} GB", device.memory_size() / (1024 * 1024 * 1024));
        println!("Unified memory: {}", device.has_unified_memory());
    }

    #[test]
    fn test_memory_topology() {
        let topology = MemoryTopology::detect();
        println!("Regions: {}", topology.regions.len());

        for region in &topology.regions {
            println!(
                "Region {:?}: {} GB, {} GB/s",
                region.id(),
                region.capacity() / (1024 * 1024 * 1024),
                region.bandwidth().0
            );
        }
    }
}
