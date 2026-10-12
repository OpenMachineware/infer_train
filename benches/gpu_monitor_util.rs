// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Utility module for GPU monitoring
// Uses mactop (https://github.com/metaspartan/mactop) to get GPU frequency

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct GpuMetrics {
    pub gpu_freq_mhz: f32,
    pub gpu_active: f32,
    pub gpu_power: f32,
    pub gpu_temp: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct MactopOutput {
    soc_metrics: SocMetrics,
    gpu_metrics: Option<GpuMetricsInner>,
}

#[derive(Debug, Clone, Deserialize)]
struct SocMetrics {
    gpu_freq_mhz: f32,
    gpu_active: f32,
    gpu_power: f32,
    gpu_temp: f32,
}

#[derive(Debug, Clone, Deserialize)]
struct GpuMetricsInner {
    freq_mhz: f32,
    active_percent: f32,
}

impl GpuMetrics {
    pub fn diff(&self, other: &GpuMetrics) -> GpuDiff {
        GpuDiff {
            freq_mhz: other.gpu_freq_mhz - self.gpu_freq_mhz,
            active_percent: other.gpu_active - self.gpu_active,
            power: other.gpu_power - self.gpu_power,
            temp: other.gpu_temp - self.gpu_temp,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuDiff {
    pub freq_mhz: f32,
    pub active_percent: f32,
    pub power: f32,
    pub temp: f32,
}

/// Get GPU metrics using mactop
/// Requires: brew install mactop
pub fn get_gpu_stats() -> Option<GpuMetrics> {
    use std::process::Command;

    let output = Command::new("mactop")
        .args(["--headless", "--format", "json", "--count", "1"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: Vec<MactopOutput> = serde_json::from_str(&stdout).ok()?;

    let first = parsed.first()?;

    Some(GpuMetrics {
        gpu_freq_mhz: first.soc_metrics.gpu_freq_mhz,
        gpu_active: first.soc_metrics.gpu_active,
        gpu_power: first.soc_metrics.gpu_power,
        gpu_temp: first.soc_metrics.gpu_temp,
    })
}

pub fn print_stats(label: &str, stats: &GpuMetrics) {
    println!("\n{}", label);
    println!("  GPU Frequency:    {:.0} MHz", stats.gpu_freq_mhz);
    println!("  GPU Active:       {:.1}%", stats.gpu_active);
    println!("  GPU Power:        {:.1} W", stats.gpu_power);
    println!("  GPU Temperature:  {:.1}°C", stats.gpu_temp);
}

pub fn print_diff(diff: &GpuDiff) {
    println!("\n--- GPU Status Changes ---");
    println!("  Frequency: {:+.0} MHz", diff.freq_mhz);
    println!("  Active:    {:+.1}%", diff.active_percent);
    println!("  Power:     {:+.1} W", diff.power);
    println!("  Temp:      {:+.1}°C", diff.temp);
}
