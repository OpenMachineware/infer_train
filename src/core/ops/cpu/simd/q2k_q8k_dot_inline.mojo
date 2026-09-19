# Q2_K × Q8_K 超优化版本
# 去掉所有包装函数，直接内联LLVM intrinsic

from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic

struct NeonU8x2(TrivialRegisterPassable):
    var lo: SIMD[DType.uint8, 16]
    var hi: SIMD[DType.uint8, 16]

struct NeonS8x2(TrivialRegisterPassable):
    var lo: SIMD[DType.int8, 16]
    var hi: SIMD[DType.int8, 16]


@always_inline
def vec_dot_q2_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q2_K × Q8_K - 激进内联，零函数调用开销"""
    var d = Float32(w_block.unsafe_offset(80).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var dmin = Float32(w_block.unsafe_offset(82).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())

    var sc_ptr = w_block.unsafe_offset(0)
    var q2_ptr = w_block.unsafe_offset(16)
    var q8_ptr = q8_data.unsafe_offset(4)
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    var m3b = SIMD[DType.uint8, 16](0x03)
    var m4b = SIMD[DType.uint8, 16](0x0F)
    var mzero = SIMD[DType.int32, 4](0)

    # Load scales+mins
    var scales_raw = sc_ptr.unsafe_load[width=16](offset=0)
    var scales_vec = scales_raw & m4b
    var mins_vec = scales_raw >> SIMD[DType.uint8, 16](4)

    # Store scales to stack
    var scales_aux = unsafe_stack_allocation[16, DType.uint8]()
    scales_aux.unsafe_store[width=16](offset=0, val=scales_vec)

    # Bias: 直接使用LLVM intrinsic
    var mins_aux = unsafe_stack_allocation[16, DType.uint8]()
    mins_aux.unsafe_store[width=16](offset=0, val=mins_vec)
    var mins_ptr = mins_aux.unsafe_bitcast[Scalar[DType.int16]]()

    # smull + add + addv 直接内联
    var mins_lo = mins_ptr.unsafe_load[width=4](offset=0)
    var mins_hi = mins_ptr.unsafe_load[width=4](offset=4)
    var mins_lo2 = mins_ptr.unsafe_load[width=4](offset=8)
    var mins_hi2 = mins_ptr.unsafe_load[width=4](offset=12)

    var bsums_ptr = q8_bsums
    var bsums_lo = bsums_ptr.unsafe_load[width=4](offset=0)
    var bsums_hi = bsums_ptr.unsafe_load[width=4](offset=4)
    var bsums_lo2 = bsums_ptr.unsafe_load[width=4](offset=8)
    var bsums_hi2 = bsums_ptr.unsafe_load[width=4](offset=12)

    # Direct smull intrinsic
    var s0 = llvm_intrinsic["llvm.aarch64.neon.smull.v4i32.v4i16", SIMD[DType.int32, 4], False](mins_lo, bsums_lo) + \
             llvm_intrinsic["llvm.aarch64.neon.smull.v4i32.v4i16", SIMD[DType.int32, 4], False](mins_hi, bsums_hi)
    var s1 = llvm_intrinsic["llvm.aarch64.neon.smull.v4i32.v4i16", SIMD[DType.int32, 4], False](mins_lo2, bsums_lo2) + \
             llvm_intrinsic["llvm.aarch64.neon.smull.v4i32.v4i16", SIMD[DType.int32, 4], False](mins_hi2, bsums_hi2)

    # Direct addv intrinsic
    var summs = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](s0 + s1)

    # Preload all q8 data
    var q8_0 = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, True](q8_ptr)
    var q8_1 = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, True](q8_ptr.unsafe_offset(32))
    var q8_2 = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, True](q8_ptr.unsafe_offset(64))
    var q8_3 = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, True](q8_ptr.unsafe_offset(96))
    var q8_4 = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, True](q8_ptr.unsafe_offset(128))
    var q8_5 = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, True](q8_ptr.unsafe_offset(160))
    var q8_6 = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, True](q8_ptr.unsafe_offset(192))
    var q8_7 = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, True](q8_ptr.unsafe_offset(224))

    var isum = Int32(0)

    # j=0: 直接内联sdot和addv
    var q2bits = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonU8x2, True](q2_ptr)

    var q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    var q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()

    # dot + addv 直接，然后乘scale
    var dot0 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_lo, q8_0.lo)
    )
    var sc0 = scales_aux.unsafe_offset(0).unsafe_load().__mlir_int()
    isum = isum + dot0 * sc0

    var dot1 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_hi, q8_0.hi)
    )
    var sc1 = scales_aux.unsafe_offset(1).unsafe_load().__mlir_int()
    isum = isum + dot1 * sc1

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    dot0 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_lo, q8_1.lo)
    )
    sc0 = scales_aux.unsafe_offset(2).unsafe_load().__mlir_int()
    isum = isum + dot0 * sc0
    dot1 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_hi, q8_1.hi)
    )
    sc1 = scales_aux.unsafe_offset(3).unsafe_load().__mlir_int()
    isum = isum + dot1 * sc1

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    dot0 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_lo, q8_2.lo)
    )
    sc0 = scales_aux.unsafe_offset(4).unsafe_load().__mlir_int()
    isum = isum + dot0 * sc0
    dot1 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_hi, q8_2.hi)
    )
    sc1 = scales_aux.unsafe_offset(5).unsafe_load().__mlir_int()
    isum = isum + dot1 * sc1

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    dot0 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_lo, q8_3.lo)
    )
    sc0 = scales_aux.unsafe_offset(6).unsafe_load().__mlir_int()
    isum = isum + dot0 * sc0
    dot1 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_hi, q8_3.hi)
    )
    sc1 = scales_aux.unsafe_offset(7).unsafe_load().__mlir_int()
    isum = isum + dot1 * sc1

    # j=1
    q2bits = llvm_intrinsic["llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonU8x2, True](q2_ptr.unsafe_offset(32))

    q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    dot0 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_lo, q8_4.lo)
    )
    sc0 = scales_aux.unsafe_offset(8).unsafe_load().__mlir_int()
    isum = isum + dot0 * sc0
    dot1 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_hi, q8_4.hi)
    )
    sc1 = scales_aux.unsafe_offset(9).unsafe_load().__mlir_int()
    isum = isum + dot1 * sc1

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    dot0 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_lo, q8_5.lo)
    )
    sc0 = scales_aux.unsafe_offset(10).unsafe_load().__mlir_int()
    isum = isum + dot0 * sc0
    dot1 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_hi, q8_5.hi)
    )
    sc1 = scales_aux.unsafe_offset(11).unsafe_load().__mlir_int()
    isum = isum + dot1 * sc1

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    dot0 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_lo, q8_6.lo)
    )
    sc0 = scales_aux.unsafe_offset(12).unsafe_load().__mlir_int()
    isum = isum + dot0 * sc0
    dot1 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_hi, q8_6.hi)
    )
    sc1 = scales_aux.unsafe_offset(13).unsafe_load().__mlir_int()
    isum = isum + dot1 * sc1

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    dot0 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_lo, q8_7.lo)
    )
    sc0 = scales_aux.unsafe_offset(14).unsafe_load().__mlir_int()
    isum = isum + dot0 * sc0
    dot1 = llvm_intrinsic["llvm.vector.reduce.add.v4i32", Int32, False](
        llvm_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4], False](mzero, q2bytes_hi, q8_7.hi)
    )
    sc1 = scales_aux.unsafe_offset(15).unsafe_load().__mlir_int()
    isum = isum + dot1 * sc1

    return d * q8_d * Float32(isum) - dmin * q8_d * Float32(summs)
