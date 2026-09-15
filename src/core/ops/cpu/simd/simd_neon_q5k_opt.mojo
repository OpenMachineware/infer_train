# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# Optimized Q5_K × Q8_K dot product kernel using NEON SIMD

from std.memory import Pointer
from std.origin import MutUntrackedOrigin

# NEON SDOT intrinsic
def neon_sdot(
    acc: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.sdot.v4i32.v16i8",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](acc, a, b)

# Helper to get scale and min for sub-block j (0-7)
def _get_scale_min_k4(j: Int, scales: Pointer[UInt8, MutUntrackedOrigin]) -> Tuple[Int, Int]:
    if j < 4:
        return (
            Int(scales.unsafe_load[width=1](offset=j)) & 63,
            Int(scales.unsafe_load[width=1](offset=j + 4)) & 63,
        )
    var d = (Int(scales.unsafe_load[width=1](offset=j + 4)) & 0xF) | (
        (Int(scales.unsafe_load[width=1](offset=j - 4)) >> 6) << 4
    )
    var m = (Int(scales.unsafe_load[width=1](offset=j + 4)) >> 4) | (
        (Int(scales.unsafe_load[width=1](offset=j)) >> 6) << 4
    )
    return (d, m)

def vec_dot_q5_k_q8_k_simd(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q5_K × Q8_K dot product using NEON SIMD - follows llama.cpp ARM implementation.
    
    Q5_K block layout (176 bytes):
    - d: fp16 scale at offset 0
    - dmin: fp16 min scale at offset 2
    - scales: 12 bytes at offset 4
    - qh: 32 bytes at offset 16 (high bits, 1 bit per element, packed)
    - qs: 128 bytes at offset 48 (low 4 bits, 256 elements packed)

    Q5_K value: 5-bit = low4 + (high_bit ? 16 : 0), range 0-31
    
    qh layout: 32 bytes = 256 bits, one bit per element
    - For element i: high_bit = (qh[i/8] >> (i%8)) & 1
    - But llama.cpp processes differently: each qh byte provides bits for 8 batches
    """
    # Read Q5_K scales
    var w_half = w_block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(w_half.unsafe_load[width=1](offset=0))
    var dmin = Float32(w_half.unsafe_load[width=1](offset=1))
    var scales_ptr = w_block.unsafe_offset(4)
    var qh = w_block.unsafe_offset(16)
    var qs = w_block.unsafe_offset(48)

    # Read Q8_K scale and data
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    # Compute bias from dmin * min term
    var bias = Float32(0)
    for j in range(8):
        var (_, m) = _get_scale_min_k4(j, scales_ptr)
        var bs0 = Int32(q8_bsums.unsafe_offset(j * 2).unsafe_load())
        var bs1 = Int32(q8_bsums.unsafe_offset(j * 2 + 1).unsafe_load())
        bias -= dmin * q8_d * Float32(m) * Float32(bs0 + bs1)

    # Load qh bits once (32 bytes = 256 bits, one bit per element)
    var qhbits_0 = qh.unsafe_load[width=16](offset=0)  # First 16 bytes (128 bits)
    var qhbits_1 = qh.unsafe_load[width=16](offset=16)  # Last 16 bytes (128 bits)
    
    # Process using SIMD - 4 groups (j=0..3), each with 64 elements
    var sumi = Int32(0)
    
    # Masks for extracting bits
    var m4b = SIMD[DType.uint8, 16](0x0F)
    var mone = SIMD[DType.uint8, 16](1)  # Mask for bit 0
    var mtwo = SIMD[DType.uint8, 16](2)  # Mask for bit 1
    
    var q5_ptr = qs
    var q8_ptr = q8_qs
    
    # Process 4 groups
    for j in range(4):
        # Load 32 bytes of qs (low 4 bits, will produce 64 elements after unpacking)
        var q5bits_0 = q5_ptr.unsafe_load[width=16](offset=0)
        var q5bits_1 = q5_ptr.unsafe_load[width=16](offset=16)
        
        # Load 64 bytes of Q8 values
        var q8bytes_0 = q8_ptr.unsafe_load[width=16](offset=0)
        var q8bytes_1 = q8_ptr.unsafe_load[width=16](offset=16)
        var q8bytes_2 = q8_ptr.unsafe_load[width=16](offset=32)
        var q8bytes_3 = q8_ptr.unsafe_load[width=16](offset=48)
        
        # Extract high bits from qh
        # Bit 0 of each qh byte provides high bit for first 32 elements
        var q5h_0 = (qhbits_0 & mone) << SIMD[DType.uint8, 16](4)
        var q5h_1 = (qhbits_1 & mone) << SIMD[DType.uint8, 16](4)
        
        # Bit 1 of each qh byte provides high bit for next 32 elements
        var q5h_2 = (qhbits_0 & mtwo) << SIMD[DType.uint8, 16](3)
        var q5h_3 = (qhbits_1 & mtwo) << SIMD[DType.uint8, 16](3)
        
        # Combine low nibble with high bit to get 5-bit value (0-31)
        var q5bytes_0 = ((q5bits_0 & m4b) | q5h_0).cast[DType.int8]()
        var q5bytes_1 = ((q5bits_1 & m4b) | q5h_1).cast[DType.int8]()
        
        # Combine high nibble with high bit
        var q5bytes_2 = ((q5bits_0 >> SIMD[DType.uint8, 16](4)) | q5h_2).cast[DType.int8]()
        var q5bytes_3 = ((q5bits_1 >> SIMD[DType.uint8, 16](4)) | q5h_3).cast[DType.int8]()
        
        # Compute dot products using SDOT
        # Scale for elements 0-31 and 32-63
        var sc_0 = Int32(_get_scale_min_k4(j * 2, scales_ptr)[0])
        var sc_1 = Int32(_get_scale_min_k4(j * 2 + 1, scales_ptr)[0])
        
        var dot_0 = neon_sdot(SIMD[DType.int32, 4](0), q5bytes_0, q8bytes_0)
        var dot_1 = neon_sdot(SIMD[DType.int32, 4](0), q5bytes_1, q8bytes_1)
        sumi += sc_0 * (dot_0.reduce_add() + dot_1.reduce_add())
        
        var dot_2 = neon_sdot(SIMD[DType.int32, 4](0), q5bytes_2, q8bytes_2)
        var dot_3 = neon_sdot(SIMD[DType.int32, 4](0), q5bytes_3, q8bytes_3)
        sumi += sc_1 * (dot_2.reduce_add() + dot_3.reduce_add())
        
        # Shift qhbits right by 2 for next iteration
        qhbits_0 = qhbits_0 >> SIMD[DType.uint8, 16](2)
        qhbits_1 = qhbits_1 >> SIMD[DType.uint8, 16](2)
        
        # Advance pointers
        q5_ptr = q5_ptr.unsafe_offset(32)
        q8_ptr = q8_ptr.unsafe_offset(64)
    
    return d * q8_d * Float32(sumi) + bias