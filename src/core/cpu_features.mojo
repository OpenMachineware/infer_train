# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/cpu_features.mojo
#
# CPU feature detection using bit masks.
# Three levels: Arch, Chip, Feature.
#
# Usage:
#   var flags = detect_cpu_flags()
#   model.cpu_flags = flags  # Store in model
#   vec_dot_q4_k_q8_k(w, x, flags)  # Pass to kernel


# ============================================================================
# Arch flags (UInt8) - CPU architecture family
# ============================================================================

comptime ARCH_ARM64_M1    = UInt8(1) << 0  # Apple M1
comptime ARCH_ARM64_M2    = UInt8(1) << 1  # Apple M2/M3/M4
comptime ARCH_ARM64_OTHER = UInt8(1) << 2  # Other ARM64
comptime ARCH_X86_AVX2    = UInt8(1) << 3  # x86 with AVX2
comptime ARCH_X86_AVX512  = UInt8(1) << 4  # x86 with AVX-512


# ============================================================================
# Chip flags (UInt8) - Specific chip characteristics
# ============================================================================

comptime CHIP_L2_4MB  = UInt8(1) << 0  # 4MB L2 (M1 Pro/Max)
comptime CHIP_L2_8MB  = UInt8(1) << 1  # 8MB L2 (M1 Ultra)
comptime CHIP_L2_16MB = UInt8(1) << 2  # 16MB L2 (M2 Max/Ultra)
comptime CHIP_L2_32MB = UInt8(1) << 3  # 32MB L2 (M3 Max)


# ============================================================================
# Feature flags (UInt32) - Optional CPU features
# ============================================================================

comptime FEATURE_NEON    = UInt32(1) << 0   # ARM NEON (all Apple Silicon)
comptime FEATURE_DOTPROD = UInt32(1) << 1   # SDOT/UDOT (all Apple Silicon)
comptime FEATURE_MMLA    = UInt32(1) << 2   # Matrix multiply (M2+)
comptime FEATURE_SVE     = UInt32(1) << 3   # Scalable Vector (not on Apple)
comptime FEATURE_FMA     = UInt32(1) << 4   # Fused multiply-add
comptime FEATURE_AVX2    = UInt32(1) << 5   # x86 AVX2
comptime FEATURE_AVX512F = UInt32(1) << 6   # x86 AVX-512F


# ============================================================================
# Combined CPU flags struct
# ============================================================================

struct CpuFlags(TrivialRegisterPassable):
    """Compact CPU flags using bit masks. Pass to kernels for dispatch."""
    var arch: UInt8
    var chip: UInt8
    var features: UInt32

    def __init__(out self):
        self.arch = UInt8(0)
        self.chip = UInt8(0)
        self.features = UInt32(0)

    def has_neon(self) -> Bool:
        return (self.features & FEATURE_NEON) != 0

    def has_mmla(self) -> Bool:
        return (self.features & FEATURE_MMLA) != 0

    def has_dotprod(self) -> Bool:
        return (self.features & FEATURE_DOTPROD) != 0

    def has_avx2(self) -> Bool:
        return (self.features & FEATURE_AVX2) != 0

    def is_m1(self) -> Bool:
        return (self.arch & ARCH_ARM64_M1) != 0

    def is_m2_plus(self) -> Bool:
        return (self.arch & ARCH_ARM64_M2) != 0


# ============================================================================
# Detection function
# ============================================================================

def detect_cpu_flags() -> CpuFlags:
    """Detect CPU flags at startup. Call once and store result."""
    var flags = CpuFlags()

    # Apple M1 defaults (most common dev machine)
    # M2/M3 detection would require sysctl calls
    flags.arch = ARCH_ARM64_M1
    flags.chip = CHIP_L2_4MB
    flags.features = FEATURE_NEON | FEATURE_DOTPROD | FEATURE_FMA
    # MMLA is M2+, not set for M1

    return flags
