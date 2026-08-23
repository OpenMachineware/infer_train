# core package - tensor runtime, devices, memory, quantization, operators.

from .device import (
    Device,
    get_default_device,
    has_metal_gpu,
    has_cuda_gpu,
    has_amd_gpu,
)
