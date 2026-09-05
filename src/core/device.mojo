# core/device.mojo
#
# Device abstraction layer and runtime accelerator detection.
#
# Mojo 1.0 note: the legacy `enum` keyword has been removed from the
# language.  Enumerations are expressed as a small register-passable struct
# carrying an `Int8` tag together with `comptime` constants.  This mirrors
# how the standard library defines `std.sys.info.Vendor`.

from std.sys.info import (
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
)


struct Device(Copyable, Equatable, ImplicitlyCopyable, Movable):
    """Backend device tag (enum-like, see module docstring)."""

    var _tag: Int8

    def __init__(out self, tag: Int8):
        self._tag = tag

    comptime CPU = Device(Int8(0))
    comptime MetalGPU = Device(Int8(1))
    comptime CUDAGPU = Device(Int8(2))
    comptime AMDGPU = Device(Int8(3))

    def __eq__(self, other: Self) -> Bool:
        return self._tag == other._tag

    def __ne__(self, other: Self) -> Bool:
        return self._tag != other._tag

    def name(self) -> String:
        if self._tag == 0:
            return String("CPU")
        if self._tag == 1:
            return String("MetalGPU")
        if self._tag == 2:
            return String("CUDAGPU")
        return String("AMDGPU")

    def __str__(self) -> String:
        return self.name()

    def is_cpu(self) -> Bool:
        return self._tag == 0

    def is_gpu(self) -> Bool:
        return self._tag != 0


def has_metal_gpu() -> Bool:
    """True when the host exposes a Metal GPU (Apple Silicon)."""
    return has_apple_gpu_accelerator()


def has_cuda_gpu() -> Bool:
    """True when the host exposes an NVIDIA GPU (reserved for later)."""
    return has_nvidia_gpu_accelerator()


def has_amd_gpu() -> Bool:
    """True when the host exposes an AMD GPU (reserved for later)."""
    return has_amd_gpu_accelerator()


def get_default_device() -> Device:
    """Return the first available accelerator, preferring Metal over CPU."""
    if has_metal_gpu():
        return Device.MetalGPU
    return Device.CPU
