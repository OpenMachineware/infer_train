# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/quantized/dequantize_fp16.mojo
#
# Dequantize Q4_K weights to FP16 for GPU inference.
# This is done once at model load time to avoid repeated dequantization.

from src.core.tensor import Tensor, tensor_zeros
from std.utils.static_tuple import StaticTuple
from std.math import abs
from .quant_types import QuantType

comptime QK_K = 256


def dequantize_q4_k_to_fp16(
    w_quant: Tensor[DType.uint8, 2],  # [N, nb * 144]
) -> Tensor[DType.float16, 2]:
    """Dequantize Q4_K weights to FP16.

    Args:
        w_quant: Quantized weights [N, nb * 144] where nb = K / 256

    Returns:
        Dequantized weights [N, K] in FP16
    """
    var N = w_quant.shape()[0]
    var total_bytes = w_quant.shape()[1]
    var nb = total_bytes // 144  # Number of 256-element blocks
    var K = nb * QK_K

    var w_fp16 = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](N, K))

    # Dequantize each row
    for row in range(N):
        var w_row = w_quant.data().unsafe_offset(row * total_bytes)
        var out_row = w_fp16.data().unsafe_offset(row * K)

        # Process each 256-element block
        for block_idx in range(nb):
            var block_ptr = w_row.unsafe_offset(block_idx * 144)
            var out_ptr = out_row.unsafe_offset(block_idx * QK_K)

            # Read block header
            var d_raw = block_ptr.unsafe_bitcast[Scalar[DType.float16]]()
            var d = Float32(d_raw.unsafe_load[width=1](offset=0))
            var dmin = Float32(d_raw.unsafe_load[width=1](offset=1))
            var scales = block_ptr.unsafe_offset(4)
            var qs_base = block_ptr.unsafe_offset(16)

            # Process 8 sub-blocks (each 32 elements)
            for sub_block in range(8):
                var is_idx = (sub_block // 2) * 2
                var il_inner = sub_block & 1

                # Extract scale and min
                var sc: UInt8
                var mn: UInt8
                if is_idx < 4:
                    sc = scales.unsafe_load[width=1](offset=is_idx) & 63
                    mn = scales.unsafe_load[width=1](offset=is_idx + 4) & 63
                else:
                    sc = UInt8(
                        (scales.unsafe_load[width=1](offset=is_idx + 4) & 0x0F)
                        | ((scales.unsafe_load[width=1](offset=is_idx - 4) & 0xC0) >> 2)
                    )
                    mn = UInt8(
                        (scales.unsafe_load[width=1](offset=is_idx + 4) >> 4)
                        | ((scales.unsafe_load[width=1](offset=is_idx) & 0xC0) >> 2)
                    )

                var dl = d * Float32(sc)
                var ml = dmin * Float32(mn)

                # Process 32 elements in this sub-block
                var q_offset = (sub_block // 2) * 32 + 16 * (sub_block & 1)
                var q_ptr = qs_base.unsafe_offset(q_offset)

                # First 16 elements (low nibble)
                for i in range(16):
                    var byte_val = q_ptr.unsafe_load[width=1](offset=i)
                    var nibble = Float32(byte_val & 0x0F)
                    var val = dl * nibble - ml
                    out_ptr.unsafe_offset(sub_block * 32 + i).unsafe_store(
                        val=Scalar[DType.float16](val)
                    )

                # Next 16 elements (high nibble)
                var dl_hi = (d / 16.0) * Float32(sc)
                for i in range(16):
                    var byte_val = q_ptr.unsafe_load[width=1](offset=i)
                    var nibble = Float32((byte_val >> 4) & 0x0F)
                    var val = dl_hi * nibble - ml
                    out_ptr.unsafe_offset(sub_block * 32 + 16 + i).unsafe_store(
                        val=Scalar[DType.float16](val)
                    )

    return w_fp16


def dequantize_weights_to_fp16(
    w_quant: Tensor[DType.uint8, 2],
    quant_type: QuantType,
) -> Tensor[DType.float16, 2]:
    """Dequantize weights to FP16 based on quantization type.

    Args:
        w_quant: Quantized weights
        quant_type: Quantization type (Q4_K_M, Q5_K, etc.)

    Returns:
        Dequantized weights in FP16
    """
    if quant_type == QuantType.Q4_K_M:
        return dequantize_q4_k_to_fp16(w_quant)
    else:
        # TODO: Implement other quantization types
        return tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](0, 0))
