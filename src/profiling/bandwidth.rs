// Bandwidth profiling for GPU kernels
// Measures actual memory bandwidth utilization vs theoretical peak

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct BandwidthStats {
    /// Kernel name
    pub kernel: String,
    /// Bytes read from memory
    pub bytes_read: u64,
    /// Bytes written to memory
    pub bytes_written: u64,
    /// Execution time in nanoseconds
    pub duration_ns: u64,
    /// Calculated bandwidth in GB/s
    pub bandwidth_gbs: f64,
    /// Bandwidth utilization (vs theoretical ~100 GB/s for M1)
    pub utilization_pct: f64,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct BandwidthProfiler {
    /// Theoretical peak bandwidth (GB/s) - M1 typical
    pub peak_bandwidth: f64,
    /// Collected statistics
    pub stats: Vec<BandwidthStats>,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl BandwidthProfiler {
    pub fn new() -> Self {
        Self {
            peak_bandwidth: 100.0, // M1 theoretical peak
            stats: Vec::new(),
        }
    }

    /// Calculate bandwidth for a kernel
    pub fn record(
        &mut self,
        kernel: &str,
        bytes_read: u64,
        bytes_written: u64,
        duration_ns: u64,
    ) {
        let total_bytes = bytes_read + bytes_written;
        let bandwidth_gbs = if duration_ns > 0 {
            (total_bytes as f64) / (duration_ns as f64) * 1e9 / 1e9
        } else {
            0.0
        };
        let utilization_pct = (bandwidth_gbs / self.peak_bandwidth) * 100.0;

        self.stats.push(BandwidthStats {
            kernel: kernel.to_string(),
            bytes_read,
            bytes_written,
            duration_ns,
            bandwidth_gbs,
            utilization_pct,
        });
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Bandwidth Utilization Summary ===");
        println!(
            "{:<30} {:>12} {:>12} {:>10} {:>8}",
            "Kernel", "Read(MB)", "Write(MB)", "BW(GB/s)", "Util%"
        );
        println!("{}", "-".repeat(75));

        for stat in &self.stats {
            println!(
                "{:<30} {:>12.2} {:>12.2} {:>10.2} {:>7.1}%",
                stat.kernel,
                stat.bytes_read as f64 / 1e6,
                stat.bytes_written as f64 / 1e6,
                stat.bandwidth_gbs,
                stat.utilization_pct,
            );
        }

        // Total
        let total_read: u64 = self.stats.iter().map(|s| s.bytes_read).sum();
        let total_write: u64 = self.stats.iter().map(|s| s.bytes_written).sum();
        let total_ns: u64 = self.stats.iter().map(|s| s.duration_ns).sum();
        let total_bw = if total_ns > 0 {
            ((total_read + total_write) as f64) / (total_ns as f64) * 1e9 / 1e9
        } else {
            0.0
        };

        println!("{}", "-".repeat(75));
        println!(
            "{:<30} {:>12.2} {:>12.2} {:>10.2} {:>7.1}%",
            "TOTAL",
            total_read as f64 / 1e6,
            total_write as f64 / 1e6,
            total_bw,
            (total_bw / self.peak_bandwidth) * 100.0,
        );
    }
}

/// Calculate memory access for GEMV
/// M = output size, K = input size
/// Read: weights (M * K bytes for F16, or compressed size for quantized) + input (K * 4 bytes)
/// Write: output (M * 4 bytes)
pub fn calc_gemv_bandwidth(
    m: usize,
    k: usize,
    weight_bytes: usize,
) -> (u64, u64) {
    let bytes_read = weight_bytes + k * 4; // weights + input
    let bytes_written = m * 4; // output
    (bytes_read as u64, bytes_written as u64)
}

/// Calculate memory access for Flash Attention decode
/// seq_len = number of KV positions
/// n_heads, head_dim = attention dimensions
/// Read: Q (n_heads * head_dim * 4) + K (seq_len * n_kv_heads * head_dim * 2) + V (same)
/// Write: output (n_heads * head_dim * 4)
pub fn calc_flash_attn_bandwidth(
    seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (u64, u64) {
    let q_bytes = n_heads * head_dim * 4; // FP32
    let kv_bytes = seq_len * n_kv_heads * head_dim * 2; // FP16 per position
    let bytes_read = q_bytes + 2 * kv_bytes; // Q + K + V
    let bytes_written = n_heads * head_dim * 4; // FP32 output
    (bytes_read as u64, bytes_written as u64)
}

/// Calculate memory access for KV cache update
/// head_dim * n_kv_heads = size per position
/// Read: input (head_dim * n_kv_heads * 4)
/// Write: cache (head_dim * n_kv_heads * 2)
pub fn calc_kv_update_bandwidth(
    head_dim: usize,
    n_kv_heads: usize,
) -> (u64, u64) {
    let bytes_read = head_dim * n_kv_heads * 4; // FP32 input
    let bytes_written = head_dim * n_kv_heads * 2; // FP16 cache
    (bytes_read as u64, bytes_written as u64)
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
pub struct BandwidthProfiler;
