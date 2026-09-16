# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/gpu/gpu_tensor.mojo
#
# GPU buffer wrapper for zero-copy activation chaining in GPU forward pass.

from max.gpu.host import DeviceContext, DeviceBuffer
from src.core.tensor import Tensor
from src.core.ops.gpu.gpu_runtime import upload, download2
from std.utils.static_tuple import StaticTuple
from std.collections.optional import Optional


# 2D version for most activations [batch, hidden]
struct GPUTensor:
    """2D tensor that can reside on CPU or GPU, with lazy transfer.

    Use this to chain multiple GPU operations without intermediate
    CPU transfers. The data stays on GPU across operations.
    """
    var cpu_data: Optional[Tensor[DType.float16, 2]]
    var gpu_buf: Optional[DeviceBuffer[DType.float16]]
    var ctx: DeviceContext
    var on_gpu: Bool
    var rows: Int
    var cols: Int

    def __init__(
        out self,
        cpu_tensor: Tensor[DType.float16, 2],
        ctx: DeviceContext,
    ):
        """Initialize with CPU tensor, GPU buffer created lazily."""
        self.cpu_data = cpu_tensor
        self.gpu_buf = None
        self.ctx = ctx
        self.on_gpu = False
        self.rows = cpu_tensor.shape()[0]
        self.cols = cpu_tensor.shape()[1]

    def __init__(
        out self,
        gpu_buffer: DeviceBuffer[DType.float16],
        ctx: DeviceContext,
        rows: Int,
        cols: Int,
    ):
        """Initialize with GPU buffer, CPU tensor created lazily."""
        self.cpu_data = None
        self.gpu_buf = gpu_buffer
        self.ctx = ctx
        self.on_gpu = True
        self.rows = rows
        self.cols = cols

    def to_gpu(mut self) raises -> DeviceBuffer[DType.float16]:
        """Get GPU buffer, uploading if necessary."""
        if not self.gpu_buf:
            if not self.cpu_data:
                raise "GPUTensor has no data"
            # Upload from CPU to GPU
            self.gpu_buf = upload[DType.float16, 2](
                self.ctx, self.cpu_data.value()
            )
            self.on_gpu = True
        return self.gpu_buf.value()

    def to_cpu(mut self) raises -> Tensor[DType.float16, 2]:
        """Get CPU tensor, downloading if necessary."""
        if not self.cpu_data:
            if not self.gpu_buf:
                raise "GPUTensor has no data"
            # Download from GPU to CPU
            self.cpu_data = download2[DType.float16](
                self.ctx,
                self.gpu_buf.value(),
                StaticTuple[Int, 2](self.rows, self.cols),
            )
            self.on_gpu = False
        return self.cpu_data.value()

    def ensure_gpu(mut self) raises:
        """Ensure data is on GPU."""
        _ = self.to_gpu()

    def ensure_cpu(mut self) raises:
        """Ensure data is on CPU."""
        _ = self.to_cpu()

    def discard_cpu(mut self):
        """Free CPU memory if data is on GPU."""
        if self.on_gpu and self.cpu_data:
            self.cpu_data = None

    def discard_gpu(mut self):
        """Free GPU memory if data is on CPU."""
        if not self.on_gpu and self.gpu_buf:
            self.gpu_buf = None

    def shape(self) -> Tuple[Int, Int]:
        return Tuple(self.rows, self.cols)

    def numel(self) -> Int:
        return self.rows * self.cols

    def is_on_gpu(self) -> Bool:
        return self.on_gpu


alias GPUBufferF16 = DeviceBuffer[DType.float16]
