# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/q4k_q8k_dot.mojo
#
# Q4_K × Q8_K int8 dot product using NEON SDOT instruction.
#
# Key optimization pattern from llama.cpp:
# 1. ld1.16b for efficient loading
# 2. sdot.4s for vector dot product
# 3. addv.4s for horizontal sum
# 4. Integer multiplication for scales AFTER horizontal sum
# 5. Float multiplication for super-block scale at the end

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic
from std.utils.static_tuple import StaticTuple
from ....tensor import Tensor, tensor_zeros

comptime QK_K = 256

# NEON intrinsics

struct NeonU8x2(TrivialRegisterPassable):
    var lo: SIMD[DType.uint8, 16]
    var hi: SIMD[DType.uint8, 16]

struct NeonS8x2(TrivialRegisterPassable):
    var lo: SIMD[DType.int8, 16]
    var hi: SIMD[DType.int8, 16]


@always_inline
def neon_ld1_u8_x2(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonU8x2:
    """Load 32 bytes using ld1.16b instruction."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonU8x2, has_side_effect=True
    ](ptr)


@always_inline
def neon_ld1_s8_x2(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonS8x2:
    """Load 32 bytes using ld1.16b instruction."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, has_side_effect=True
    ](ptr)


@always_inline
def neon_sdot(
    acc: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    """NEON SDOT: int8 × int8 -> int32 dot product."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.sdot.v4i32.v16i8",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](acc, a, b)


@always_inline
def neon_addv(v: SIMD[DType.int32, 4]) -> Int32:
    """Horizontal sum using addv.4s - LLVM intrinsic."""
    return llvm_intrinsic[
        "llvm.vector.reduce.add.v4i32",
        Int32,
        has_side_effect=False,
    ](v)


@always_inline
def vec_dot_q4_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q4_K × Q8_K dot product optimized with llama.cpp pattern.

    Q4_K block layout (144 bytes):
    - d: fp16 scale at offset 0
    - dmin: fp16 min scale at offset 2
    - scales: 12 bytes at offset 4
    - qs: 128 bytes at offset 16 (4-bit values, 256 elements packed)

    Q8_K layout (292 bytes):
    - d: float32 scale at offset 0
    - qs: 256 int8 at offset 4
    - bsums: 16 int16 at offset 260
    """
    var mzero = SIMD[DType.int32, 4](0)
    var m4b = SIMD[DType.uint8, 16](0x0F)

    # Load super-block scales
    var d = Float32(w_block.unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var dmin = Float32(w_block.unsafe_offset(2).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())

    # Pre-compute scales (like llama.cpp)
    var scales_raw = w_block.unsafe_offset(4)
    var sc0 = Int(scales_raw.unsafe_load[width=1](offset=0).value()) & 0x3F
    var sc1 = Int(scales_raw.unsafe_load[width=1](offset=1).value()) & 0x3F
    var sc2 = Int(scales_raw.unsafe_load[width=1](offset=2).value()) & 0x3F
    var sc3 = Int(scales_raw.unsafe_load[width=1](offset=3).value()) & 0x3F
    var sc4 = (Int(scales_raw.unsafe_load[width=1](offset=8).value()) & 0x0F) | \
              ((Int(scales_raw.unsafe_load[width=1](offset=0).value()) >> 6) << 4)
    var sc5 = (Int(scales_raw.unsafe_load[width=1](offset=9).value()) & 0x0F) | \
              ((Int(scales_raw.unsafe_load[width=1](offset=1).value()) >> 6) << 4)
    var sc6 = (Int(scales_raw.unsafe_load[width=1](offset=10).value()) & 0x0F) | \
              ((Int(scales_raw.unsafe_load[width=1](offset=2).value()) >> 6) << 4)
    var sc7 = (Int(scales_raw.unsafe_load[width=1](offset=11).value()) & 0x0F) | \
              ((Int(scales_raw.unsafe_load[width=1](offset=3).value()) >> 6) << 4)

    # Pre-compute min values for bias
    var m0 = Int(scales_raw.unsafe_load[width=1](offset=4).value()) & 0x3F
    var m1 = Int(scales_raw.unsafe_load[width=1](offset=5).value()) & 0x3F
    var m2 = Int(scales_raw.unsafe_load[width=1](offset=6).value()) & 0x3F
    var m3 = Int(scales_raw.unsafe_load[width=1](offset=7).value()) & 0x3F
    var m4 = (Int(scales_raw.unsafe_load[width=1](offset=12).value()) & 0x0F) | \
             ((Int(scales_raw.unsafe_load[width=1](offset=4).value()) >> 6) << 4)
    var m5 = (Int(scales_raw.unsafe_load[width=1](offset=13).value()) & 0x0F) | \
             ((Int(scales_raw.unsafe_load[width=1](offset=5).value()) >> 6) << 4)
    var m6 = (Int(scales_raw.unsafe_load[width=1](offset=14).value()) & 0x0F) | \
             ((Int(scales_raw.unsafe_load[width=1](offset=6).value()) >> 6) << 4)
    var m7 = (Int(scales_raw.unsafe_load[width=1](offset=15).value()) & 0x0F) | \
             ((Int(scales_raw.unsafe_load[width=1](offset=7).value()) >> 6) << 4)

    # Compute bias from Q8_K bsums
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    var bias = Float32(0)

    # bsums covers 16 elements each, scales cover 32 elements each
    # For each scale j, we need bsums[j*2] and bsums[j*2+1]
    var bsum0 = Int32(q8_bsums.unsafe_offset(0).unsafe_load()) + Int32(q8_bsums.unsafe_offset(1).unsafe_load())
    var bsum1 = Int32(q8_bsums.unsafe_offset(2).unsafe_load()) + Int32(q8_bsums.unsafe_offset(3).unsafe_load())
    var bsum2 = Int32(q8_bsums.unsafe_offset(4).unsafe_load()) + Int32(q8_bsums.unsafe_offset(5).unsafe_load())
    var bsum3 = Int32(q8_bsums.unsafe_offset(6).unsafe_load()) + Int32(q8_bsums.unsafe_offset(7).unsafe_load())
    var bsum4 = Int32(q8_bsums.unsafe_offset(8).unsafe_load()) + Int32(q8_bsums.unsafe_offset(9).unsafe_load())
    var bsum5 = Int32(q8_bsums.unsafe_offset(10).unsafe_load()) + Int32(q8_bsums.unsafe_offset(11).unsafe_load())
    var bsum6 = Int32(q8_bsums.unsafe_offset(12).unsafe_load()) + Int32(q8_bsums.unsafe_offset(13).unsafe_load())
    var bsum7 = Int32(q8_bsums.unsafe_offset(14).unsafe_load()) + Int32(q8_bsums.unsafe_offset(15).unsafe_load())

    bias = dmin * q8_d * Float32(
        Int32(m0) * bsum0 + Int32(m1) * bsum1 + Int32(m2) * bsum2 + Int32(m3) * bsum3 +
        Int32(m4) * bsum4 + Int32(m5) * bsum5 + Int32(m6) * bsum6 + Int32(m7) * bsum7
    )

    # Base pointers
    var q4_ptr = w_block.unsafe_offset(16)
    var q8_ptr = q8_data.unsafe_offset(4)

    var sumi1 = Int32(0)
    var sumi2 = Int32(0)

    # Unrolled inner loop: j=0
    var q4bits = neon_ld1_u8_x2(q4_ptr)
    var q8bytes = neon_ld1_s8_x2(q8_ptr)
    var p1 = neon_sdot(neon_sdot(mzero, (q4bits.lo & m4b).cast[DType.int8](), q8bytes.lo),
                       (q4bits.hi & m4b).cast[DType.int8](), q8bytes.hi)
    sumi1 += Int32(neon_addv(p1) * sc0)

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(32))
    var p2 = neon_sdot(neon_sdot(mzero, (q4bits.lo >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.lo),
                       (q4bits.hi >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.hi)
    sumi2 += Int32(neon_addv(p2) * sc4)

    # j=1
    q4bits = neon_ld1_u8_x2(q4_ptr.unsafe_offset(32))
    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(64))
    p1 = neon_sdot(neon_sdot(mzero, (q4bits.lo & m4b).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi & m4b).cast[DType.int8](), q8bytes.hi)
    sumi1 += Int32(neon_addv(p1) * sc1)

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(96))
    p2 = neon_sdot(neon_sdot(mzero, (q4bits.lo >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.hi)
    sumi2 += Int32(neon_addv(p2) * sc5)

    # j=2
    q4bits = neon_ld1_u8_x2(q4_ptr.unsafe_offset(64))
    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(128))
    p1 = neon_sdot(neon_sdot(mzero, (q4bits.lo & m4b).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi & m4b).cast[DType.int8](), q8bytes.hi)
    sumi1 += Int32(neon_addv(p1) * sc2)

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(160))
    p2 = neon_sdot(neon_sdot(mzero, (q4bits.lo >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.hi)
    sumi2 += Int32(neon_addv(p2) * sc6)

    # j=3
    q4bits = neon_ld1_u8_x2(q4_ptr.unsafe_offset(96))
    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(192))
    p1 = neon_sdot(neon_sdot(mzero, (q4bits.lo & m4b).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi & m4b).cast[DType.int8](), q8bytes.hi)
    sumi1 += Int32(neon_addv(p1) * sc3)

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(224))
    p2 = neon_sdot(neon_sdot(mzero, (q4bits.lo >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.hi)
    sumi2 += Int32(neon_addv(p2) * sc7)

    # Apply super-block scale
    return d * q8_d * Float32(sumi1 + sumi2) - bias


def matmul_q4_k_q8_k_row(
    w_quant: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
    N: Int,
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Tensor[DType.float16, 1]:
    """Compute one row of Q4_K × Q8_K matmul."""
    var out = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](N))

    for n in range(N):
        var sumf = Float32(0)
        for b in range(nb):
            var row_offset = n * nb * 144 + b * 144
            var val = vec_dot_q4_k_q8_k(
                w_quant.unsafe_offset(row_offset),
                q8_data.unsafe_offset(b * 292),
            )
            sumf += val
        out.data().unsafe_offset(n).unsafe_store(val=Scalar[DType.float16](sumf))

    return out
