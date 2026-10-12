// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// CPU capabilities detection
// Detects architecture, features, and supported instructions

use std::fmt;

/// CPU architecture type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    Arm,
    X86,
    Unknown,
}

/// ARM-specific features
#[derive(Debug, Clone, Copy, Default)]
pub struct ArmFeatures {
    /// NEON SIMD (baseline for ARMv8)
    pub neon: bool,
    /// Dot product instructions (vdot, ARMv8.1+)
    pub dotprod: bool,
    /// FP16 arithmetic (ARMv8.2+)
    pub fp16: bool,
    /// BF16 arithmetic (ARMv8.6+ / v9)
    pub bf16: bool,
    /// SVE (Scalable Vector Extension)
    pub sve: bool,
    /// SVE2
    pub sve2: bool,
    /// i8mm (Int8 matrix multiply)
    pub i8mm: bool,
}

/// x86-specific features
#[derive(Debug, Clone, Copy, Default)]
pub struct X86Features {
    /// SSE4.1
    pub sse41: bool,
    /// SSE4.2
    pub sse42: bool,
    /// AVX (256-bit)
    pub avx: bool,
    /// AVX2
    pub avx2: bool,
    /// AVX-512F
    pub avx512f: bool,
    /// AVX-512VL
    pub avx512vl: bool,
    /// AVX-512BW
    pub avx512bw: bool,
    /// AVX-512DQ
    pub avx512dq: bool,
    /// AVX-512VNNI (Int8 dot product)
    pub avx512vnni: bool,
    /// AMX (Advanced Matrix Extensions)
    pub amx: bool,
}

/// Detected CPU capabilities
#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    /// Architecture (ARM or x86)
    pub arch: CpuArch,
    /// CPU vendor (Apple, Intel, AMD, etc.)
    pub vendor: String,
    /// CPU model name
    pub model: String,
    /// Number of physical cores
    pub cores: usize,
    /// L1 data cache size (bytes)
    pub l1_data_cache: usize,
    /// L2 cache size (bytes)
    pub l2_cache: usize,
    /// ARM-specific features
    pub arm: ArmFeatures,
    /// x86-specific features
    pub x86: X86Features,
}

impl Default for CpuCapabilities {
    fn default() -> Self {
        Self {
            arch: CpuArch::Unknown,
            vendor: String::new(),
            model: String::new(),
            cores: 1,
            l1_data_cache: 32 * 1024, // 32KB default
            l2_cache: 256 * 1024,     // 256KB default
            arm: ArmFeatures::default(),
            x86: X86Features::default(),
        }
    }
}

impl fmt::Display for CpuCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} ({} cores", self.vendor, self.model, self.cores)?;
        match self.arch {
            CpuArch::Arm => {
                if self.arm.dotprod {
                    write!(f, ", NEON+dotprod")?;
                } else if self.arm.neon {
                    write!(f, ", NEON")?;
                }
            }
            CpuArch::X86 => {
                if self.x86.avx512f {
                    write!(f, ", AVX-512")?;
                } else if self.x86.avx2 {
                    write!(f, ", AVX2")?;
                } else if self.x86.avx {
                    write!(f, ", AVX")?;
                }
            }
            CpuArch::Unknown => {}
        }
        write!(f, ")")
    }
}

/// Detect CPU capabilities
pub fn detect() -> CpuCapabilities {
    #[cfg(target_arch = "aarch64")]
    {
        detect_arm()
    }
    #[cfg(target_arch = "x86_64")]
    {
        detect_x86()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        CpuCapabilities::default()
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_arm() -> CpuCapabilities {
    use std::arch::aarch64::*;

    let mut caps = CpuCapabilities { arch: CpuArch::Arm, ..Default::default() };

    // Get CPU info via sysctl on macOS
    #[cfg(target_os = "macos")]
    {
        caps.vendor = "Apple".to_string();
        caps.model = get_sysctl_string("machdep.cpu.brand_string")
            .unwrap_or_else(|| "Apple Silicon".to_string());
        caps.cores = get_sysctl_int("hw.physicalcpu").unwrap_or(1) as usize;
        caps.l1_data_cache =
            get_sysctl_int("hw.l1dcachesize").unwrap_or(32 * 1024) as usize;
        caps.l2_cache =
            get_sysctl_int("hw.l2cachesize").unwrap_or(256 * 1024) as usize;
    }

    #[cfg(target_os = "linux")]
    {
        caps.vendor = "ARM".to_string();
        caps.model = "Unknown ARM".to_string();
        caps.cores = num_cpus::get_physical();
    }

    // Detect features via ID registers (ARMv8.1+)
    // Note: On macOS, these are readable but may require special handling
    unsafe {
        // NEON is always available on ARMv8
        caps.arm.neon = true;

        // Try to detect dotprod via ID_AA64ISAR0_EL1
        // On Apple M1+, dotprod is always available
        // For simplicity, assume available on ARMv8
        caps.arm.dotprod = true; // M1/M2/M3 have dotprod

        // FP16 arithmetic - available on M2+
        // For now, check if we're on a newer Apple Silicon
        if caps.model.contains("M2")
            || caps.model.contains("M3")
            || caps.model.contains("M4")
        {
            caps.arm.fp16 = true;
            caps.arm.bf16 = true;
        }
    }

    caps
}

#[cfg(target_arch = "x86_64")]
fn detect_x86() -> CpuCapabilities {
    let mut caps = CpuCapabilities { arch: CpuArch::X86, ..Default::default() };

    // Use is_x86_feature_detected! macros
    caps.x86.sse41 = is_x86_feature_detected!("sse4.1");
    caps.x86.sse42 = is_x86_feature_detected!("sse4.2");
    caps.x86.avx = is_x86_feature_detected!("avx");
    caps.x86.avx2 = is_x86_feature_detected!("avx2");
    caps.x86.avx512f = is_x86_feature_detected!("avx512f");
    caps.x86.avx512vl = is_x86_feature_detected!("avx512vl");
    caps.x86.avx512bw = is_x86_feature_detected!("avx512bw");
    caps.x86.avx512dq = is_x86_feature_detected!("avx512dq");

    // Get CPU info
    #[cfg(target_os = "macos")]
    {
        caps.vendor = get_sysctl_string("machdep.cpu.vendor")
            .unwrap_or_else(|| "Unknown".to_string());
        caps.model = get_sysctl_string("machdep.cpu.brand_string")
            .unwrap_or_else(|| "Unknown x86".to_string());
        caps.cores = get_sysctl_int("hw.physicalcpu").unwrap_or(1) as usize;
    }

    #[cfg(target_os = "linux")]
    {
        caps.vendor = "Unknown".to_string();
        caps.model = "Unknown x86".to_string();
        caps.cores = num_cpus::get_physical();
    }

    caps
}

#[cfg(target_os = "macos")]
fn get_sysctl_string(name: &str) -> Option<String> {
    use std::ffi::CStr;
    use std::ptr;

    let mut size: usize = 0;
    let name_c = std::ffi::CString::new(name).ok()?;

    unsafe {
        // Get size first
        if libc::sysctlbyname(
            name_c.as_ptr(),
            ptr::null_mut(),
            &mut size,
            ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }

        // Allocate buffer
        let mut buf = vec![0u8; size];
        if libc::sysctlbyname(
            name_c.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }

        CStr::from_bytes_with_nul(&buf)
            .ok()
            .and_then(|c| c.to_str().ok())
            .map(|s| s.to_string())
    }
}

#[cfg(target_os = "macos")]
fn get_sysctl_int(name: &str) -> Option<i64> {
    use std::ffi::CString;
    use std::ptr;

    let name_c = CString::new(name).ok()?;
    let mut value: i64 = 0;
    let mut size = std::mem::size_of::<i64>();

    unsafe {
        if libc::sysctlbyname(
            name_c.as_ptr(),
            &mut value as *mut i64 as *mut libc::c_void,
            &mut size,
            ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }

    Some(value)
}
