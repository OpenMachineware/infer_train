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

from std.memory import Pointer, unsafe_stack_allocation
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
def neon_addp(a: SIMD[DType.int16, 8], b: SIMD[DType.int16, 8]) -> SIMD[DType.int16, 8]:
    """Pairwise add using addp.8h."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.addp.v8i16",
        SIMD[DType.int16, 8],
        has_side_effect=False,
    ](a, b)


@always_inline
def neon_smull(a: SIMD[DType.int16, 4], b: SIMD[DType.int16, 4]) -> SIMD[DType.int32, 4]:
    """Widening multiply using NEON smull instruction."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.smull.v4i32",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](a, b)


@always_inline
def neon_get_low(v: SIMD[DType.int16, 8]) -> SIMD[DType.int16, 4]:
    """Extract low half of int16x8 (like vget_low_s16)."""
    return SIMD[DType.int16, 4](v[0], v[1], v[2], v[3])


@always_inline
def neon_get_high(v: SIMD[DType.int16, 8]) -> SIMD[DType.int16, 4]:
    """Extract high half of int16x8 (like vget_high_s16)."""
    return SIMD[DType.int16, 4](v[4], v[5], v[6], v[7])


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

    # Decode scales following llama.cpp exactly
    # Load 12 bytes (3 uint32) - using individual loads to avoid overfetching
    var scales_raw_ptr = w_block.unsafe_offset(4).unsafe_bitcast[Scalar[DType.uint32]]()
    var utmp0 = UInt32(scales_raw_ptr.unsafe_offset(0).unsafe_load())
    var utmp1 = UInt32(scales_raw_ptr.unsafe_offset(1).unsafe_load())
    var utmp2 = UInt32(scales_raw_ptr.unsafe_offset(2).unsafe_load())

    var kmask1 = UInt32(0x3f3f3f3f)
    var kmask2 = UInt32(0x0f0f0f0f)
    var kmask3 = UInt32(0x03030303)

    # Extract mins8 (2 uint32 packed into 8 bytes)
    # mins8[0] = utmp[1] & kmask1
    # mins8[1] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4)
    var mins8_0 = utmp1 & kmask1
    var mins8_1 = ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4)

    # Repack scales in place (following llama.cpp exactly)
    # utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4)
    # utmp[0] &= kmask1
    var new_utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4)
    var new_utmp0 = utmp0 & kmask1

    # Extract scale values from repacked utmp (like llama.cpp: const uint8_t * scales = (const uint8_t *)utmp)
    # scales[0-3] from new_utmp0, scales[4-7] from new_utmp1
    # Using Int32 for multiplication with dot product result
    var sc0 = Int32(new_utmp0 & 0xFF)
    var sc1 = Int32((new_utmp0 >> 8) & 0xFF)
    var sc2 = Int32((new_utmp0 >> 16) & 0xFF)
    var sc3 = Int32((new_utmp0 >> 24) & 0xFF)
    var sc4 = Int32(new_utmp1 & 0xFF)
    var sc5 = Int32((new_utmp1 >> 8) & 0xFF)
    var sc6 = Int32((new_utmp1 >> 16) & 0xFF)
    var sc7 = Int32((new_utmp1 >> 24) & 0xFF)

    # Extract mins8 as 2 uint32 values (like llama.cpp: uint32x2_t mins8)
    # mins8[0] = utmp[1] & kmask1
    # mins8[1] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4)
    var mins8_lo = mins8_0
    var mins8_hi = mins8_1

    # Store to stack and load as uint8x8 (like vreinterpret_u8_u32)
    var mins8_stack = unsafe_stack_allocation[8, DType.uint8]()
    mins8_stack.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(offset=0, val=mins8_lo)
    mins8_stack.unsafe_bitcast[Scalar[DType.uint32]]().unsafe_store(offset=1, val=mins8_hi)
    var mins8_bytes = mins8_stack.unsafe_load[width=8]()

    # Widen to int16x8 using vector widening (like vmovl_u8 + vreinterpret)
    # Direct cast maps to LLVM zext which becomes uxtl.8h
    var mins = mins8_bytes.cast[DType.int16]()

    # Compute bias using NEON intrinsics (like llama.cpp)
    # Load bsums as int16 vectors using SIMD unsafe_load (generates ld1.8h)
    var q8_bsums_ptr = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    var bsums_lo = q8_bsums_ptr.unsafe_load[width=8]()
    var bsums_hi = q8_bsums_ptr.unsafe_load[width=8](offset=8)

    # Pairwise add (vpaddq_s16) - use NEON intrinsic
    var q8sums = neon_addp(bsums_lo, bsums_hi)

    # Widening multiply: int16 * int16 -> int32 (like vmull_s16)
    # llama.cpp: vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins))
    # Both low and high halves need to be extracted as int16x4
    var prod_lo = neon_smull(neon_get_low(q8sums), neon_get_low(mins))
    var prod_hi = neon_smull(neon_get_high(q8sums), neon_get_high(mins))
    var prod = prod_lo + prod_hi

    # Horizontal sum (like vaddvq_s32)
    var summs = neon_addv(prod)

    var bias = dmin * q8_d * Float32(summs)

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
    sumi1 += neon_addv(p1) * sc0  # scales[2*0+0]

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(32))
    var p2 = neon_sdot(neon_sdot(mzero, (q4bits.lo >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.lo),
                       (q4bits.hi >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.hi)
    sumi2 += neon_addv(p2) * sc1  # scales[2*0+1]

    # j=1
    q4bits = neon_ld1_u8_x2(q4_ptr.unsafe_offset(32))
    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(64))
    p1 = neon_sdot(neon_sdot(mzero, (q4bits.lo & m4b).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi & m4b).cast[DType.int8](), q8bytes.hi)
    sumi1 += neon_addv(p1) * sc2  # scales[2*1+0]

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(96))
    p2 = neon_sdot(neon_sdot(mzero, (q4bits.lo >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.hi)
    sumi2 += neon_addv(p2) * sc3  # scales[2*1+1]

    # j=2
    q4bits = neon_ld1_u8_x2(q4_ptr.unsafe_offset(64))
    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(128))
    p1 = neon_sdot(neon_sdot(mzero, (q4bits.lo & m4b).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi & m4b).cast[DType.int8](), q8bytes.hi)
    sumi1 += neon_addv(p1) * sc4  # scales[2*2+0]

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(160))
    p2 = neon_sdot(neon_sdot(mzero, (q4bits.lo >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.hi)
    sumi2 += neon_addv(p2) * sc5  # scales[2*2+1]

    # j=3
    q4bits = neon_ld1_u8_x2(q4_ptr.unsafe_offset(96))
    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(192))
    p1 = neon_sdot(neon_sdot(mzero, (q4bits.lo & m4b).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi & m4b).cast[DType.int8](), q8bytes.hi)
    sumi1 += neon_addv(p1) * sc6  # scales[2*3+0]

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(224))
    p2 = neon_sdot(neon_sdot(mzero, (q4bits.lo >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.lo),
                   (q4bits.hi >> SIMD[DType.uint8, 16](4)).cast[DType.int8](), q8bytes.hi)
    sumi2 += neon_addv(p2) * sc7  # scales[2*3+1]

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
