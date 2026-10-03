// Get GPU status (utilization, temperature) before/after benchmarks
// Uses IOKit to query Apple GPU

use metal::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct GpuStats {
    device_utilization: f32,
    renderer_utilization: f32,
    tiler_utilization: f32,
    alloc_system_memory: u64,
    in_use_system_memory: u64,
}

fn parse_performance_stats(output: &str) -> Option<GpuStats> {
    // Find the PerformanceStatistics line
    let start = output.find("\"PerformanceStatistics\" = {")?;
    let end = output[start..].find("}").map(|i| start + i + 1)?;

    let stats_str = &output[start..end];

    // Parse the values
    fn extract_value(s: &str, key: &str) -> Option<f32> {
        let pattern = format!("\"{}\"=", key);
        let start = s.find(&pattern)?;
        let value_str = &s[start + pattern.len()..];
        let end = value_str
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(value_str.len());
        value_str[..end].parse().ok()
    }

    fn extract_int(s: &str, key: &str) -> Option<u64> {
        let pattern = format!("\"{}\"=", key);
        let start = s.find(&pattern)?;
        let value_str = &s[start + pattern.len()..];
        let end = value_str
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(value_str.len());
        value_str[..end].parse().ok()
    }

    Some(GpuStats {
        device_utilization: extract_value(stats_str, "Device Utilization %")?,
        renderer_utilization: extract_value(
            stats_str,
            "Renderer Utilization %",
        )?,
        tiler_utilization: extract_value(stats_str, "Tiler Utilization %")?,
        alloc_system_memory: extract_int(stats_str, "Alloc system memory")?,
        in_use_system_memory: extract_int(stats_str, "In use system memory")?,
    })
}

fn get_gpu_stats_from_iokit() -> Option<GpuStats> {
    use std::process::Command;

    let output = Command::new("ioreg").args(["-l", "-w0"]).output().ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_performance_stats(&stdout)
}

fn print_stats(label: &str, stats: &GpuStats) {
    println!("\n{}", label);
    println!("  Device Utilization:  {:.1}%", stats.device_utilization);
    println!("  Renderer Utilization: {:.1}%", stats.renderer_utilization);
    println!("  Tiler Utilization:    {:.1}%", stats.tiler_utilization);
    println!(
        "  Alloc System Memory:  {:.1} MB",
        stats.alloc_system_memory as f64 / 1024.0 / 1024.0
    );
    println!(
        "  In Use System Memory: {:.1} MB",
        stats.in_use_system_memory as f64 / 1024.0 / 1024.0
    );
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    println!("=== GPU Status Monitor ===");
    println!("Device: {}", device.name());

    // Before benchmark
    let before = get_gpu_stats_from_iokit().expect("Failed to get GPU stats");
    print_stats("Before benchmark:", &before);

    println!("\n--- Running workload (100 iterations) ---");

    // Run a simple kernel to warm up GPU
    let shader = r#"
#include <metal_stdlib>
using namespace metal;

kernel void warmup(device float *out [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
    float sum = 0.0;
    for (int i = 0; i < 1000; i++) {
        sum += sin(float(i)) * cos(float(i));
    }
    out[idx] = sum;
}
"#;

    let lib =
        device.new_library_with_source(shader, &CompileOptions::new()).unwrap();
    let func = lib.get_function("warmup", None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    let buf =
        device.new_buffer(1024 * 4, MTLResourceOptions::StorageModeShared);

    for _ in 0..100 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // After workload
    let after = get_gpu_stats_from_iokit().expect("Failed to get GPU stats");
    print_stats("After workload:", &after);

    // Diff
    println!("\n--- Changes ---");
    println!(
        "  Device Utilization:  {:.1}% -> {:.1}% ({:+.1}%)",
        before.device_utilization,
        after.device_utilization,
        after.device_utilization - before.device_utilization
    );
    println!(
        "  Renderer Utilization: {:.1}% -> {:.1}% ({:+.1}%)",
        before.renderer_utilization,
        after.renderer_utilization,
        after.renderer_utilization - before.renderer_utilization
    );
    println!(
        "  Alloc System Memory:  {:.1} MB -> {:.1} MB ({:+.1} MB)",
        before.alloc_system_memory as f64 / 1024.0 / 1024.0,
        after.alloc_system_memory as f64 / 1024.0 / 1024.0,
        (after.alloc_system_memory as i64 - before.alloc_system_memory as i64)
            as f64
            / 1024.0
            / 1024.0
    );

    println!("\n=== Notes ===");
    println!("Apple Silicon GPU (M1 Max) typical specs:");
    println!("  - Max frequency: ~1296 MHz");
    println!("  - Cores: 32");
    println!("  - Frequency is dynamically adjusted based on thermal/workload");
    println!("\nFor accurate benchmarks:");
    println!("  1. Add cooldown periods between tests");
    println!("  2. Run alternating A-B-A-B tests to cancel thermal effects");
    println!(
        "  3. Monitor changes in utilization to detect thermal throttling"
    );
}
