# Test NEON TBL intrinsic for IQ4_XS table lookup
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic

# Non-linear quantization values for IQ4 formats
comptime kvalues_iq4nl = SIMD[DType.int8, 16](
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
)

@always_inline
def neon_tbl1(table: SIMD[DType.int8, 16], indices: SIMD[DType.uint8, 16]) -> SIMD[DType.int8, 16]:
    """NEON TBL1: table lookup for 16 values from 16-entry table."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.tbl1.v16i8",
        SIMD[DType.int8, 16],
        has_side_effect=False,
    ](table, indices)

def main():
    # Test: look up values from kvalues_iq4nl
    var table = kvalues_iq4nl

    # Indices: 0, 1, 2, ..., 15
    var indices = SIMD[DType.uint8, 16](
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    )

    var result = neon_tbl1(table, indices)

    print("Table: ", table)
    print("Indices: ", indices)
    print("Result: ", result)

    # Should get the same values as the table
    # result[i] should equal table[indices[i]]
