# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/rms_norm_q8k_fused.mojo
#
# Fused RMSNorm + Q8_K quantization kernel.
#
# Instead of:
#   normed = rms_norm(x)  # allocate new tensor
#   q8k = quantize_q8k(normed)  # read normed again
#
# We do:
#   q8k = rms_norm_to_q8k(x)  # compute norm and quantize in one pass

from src.core.tensor import Tensor, tensor_zeros
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.math import sqrt, abs

comptime QK_K = 256


def rms_norm_to_q8k(
    x: Tensor[DType.float16, 2],
    weight: Tensor[DType.float16, 1],
    eps: Float32,
    dst: Pointer[UInt8, MutUntrackedOrigin],
):
    """Fused RMSNorm + Q8_K quantization.

    Computes RMSNorm on x and directly quantizes the result to Q8_K format.
    This avoids allocating an intermediate fp16 tensor for the normalized values.

    Args:
        x: Input tensor [M, K]
        weight: RMSNorm weight [K]
        eps: Epsilon for numerical stability
        dst: Output buffer for Q8_K quantized data (M * nb * 292 bytes)

    Layout:
        For each row:
        - Compute RMSNorm: normed = x / sqrt(mean(x^2) + eps) * weight
        - Quantize to Q8_K: find max of normalized values, compute scale, quantize to int8
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var nb = K // QK_K

    # Allocate temporary buffer for normalized values (one block at a time)
    var normed_buf = unsafe_alloc[Float32](QK_K)

    comptime W = 8
    var d_main = (K // W) * W

    for row in range(M):
        var row_offset = row * K

        # Step 1: Compute RMSNorm variance for the entire row
        var ss = Float32(0)
        var j = 0
        while j < d_main:
            var v = x.data().unsafe_load[width=W](offset=row_offset + j)
            var v_f32 = v.cast[DType.float32]()
            ss = ss + (v_f32 * v_f32).reduce_add()
            j += W

        # Handle remaining elements
        while j < K:
            var v = Float32(x.get(row_offset + j))
            ss += v * v
            j += 1

        # Compute RMSNorm inverse for the entire row
        var r = sqrt(ss / Float32(K) + eps)
        var inv_rms = Float32(1) / r

        # Process each QK_K block
        for b in range(nb):
            var block_start = b * QK_K
            var block_dst = dst.unsafe_offset((row * nb + b) * 292)

            # Step 2: Compute normalized values for this block and find max
            var amax = Float32(0)
            var block_d_main = (QK_K // W) * W
            j = 0
            while j < block_d_main:
                var v = x.data().unsafe_load[width=W](offset=row_offset + block_start + j)
                var w = weight.data().unsafe_load[width=W](offset=block_start + j)
                var v_f32 = v.cast[DType.float32]()
                var w_f32 = w.cast[DType.float32]()
                var normed = v_f32 * SIMD[DType.float32, W](inv_rms) * w_f32

                # Store to temp buffer
                for k in range(W):
                    normed_buf.unsafe_offset(j + k).unsafe_store(val=normed[k])

                # Find max
                for k in range(W):
                    var ax = abs(normed[k])
                    if ax > amax:
                        amax = ax
                j += W

            # Handle remaining elements
            while j < QK_K:
                var v = Float32(x.get(row_offset + block_start + j))
                var w = Float32(weight.get(block_start + j))
                var normed = v * inv_rms * w
                normed_buf.unsafe_offset(j).unsafe_store(val=normed)
                var ax = abs(normed)
                if ax > amax:
                    amax = ax
                j += 1

            # Handle zero case
            if amax == 0:
                block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
                    val=Scalar[DType.float32](0)
                )
                continue

            # Compute Q8_K scale
            var iscale = 127.0 / amax
            var d = amax / 127.0

            # Store scale
            block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(
                val=Scalar[DType.float32](d)
            )

            # Step 3: Quantize to int8
            var qs_ptr = block_dst.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()

            for j in range(QK_K):
                var normed = normed_buf.unsafe_offset(j).unsafe_load()
                var qv = Int(round(iscale * normed))
                if qv > 127:
                    qv = 127
                if qv < -127:
                    qv = -127
                qs_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int8](qv))

            # Compute bsums (sums of 16-element groups)
            var bsums_ptr = block_dst.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
            for g in range(16):
                var sum = Int16(0)
                for k in range(16):
                    sum += Int16(qs_ptr.unsafe_offset(g * 16 + k).unsafe_load())
                bsums_ptr.unsafe_offset(g).unsafe_store(val=Scalar[DType.int16](sum))

    normed_buf.unsafe_free()
