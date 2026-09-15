# Test Q5_K kernel independently
from src.core.ops.cpu.simd.simd_neon import vec_dot_q5_k_q8_k
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc

def main():
    # Create test Q5_K block (176 bytes)
    # Layout: d(2) dmin(2) scales(12) qh(32) qs(128)
    var q5_block = unsafe_alloc[UInt8](176)
    
    # Create test Q8_K activation (292 bytes)
    # Layout: d(4) qs(256) bsums(16)
    var q8_data = unsafe_alloc[UInt8](292)
    
    # Initialize with simple test data
    # Set d = 1.0 (fp16)
    var d_half = q5_block.unsafe_bitcast[Scalar[DType.float16]]()
    d_half.unsafe_store(0, 1.0)
    
    # Set dmin = 0 (fp16)
    d_half.unsafe_store(1, 0.0)
    
    # Set scales to 1 (all 12 bytes)
    for i in range(12):
        q5_block.unsafe_offset(4 + i).unsafe_store(0, 1)
    
    # Set qh to 0 (no high bits)
    for i in range(32):
        q5_block.unsafe_offset(16 + i).unsafe_store(0, 0)
    
    # Set qs to simple pattern (low 4 bits = 5)
    for i in range(128):
        q5_block.unsafe_offset(48 + i).unsafe_store(0, 0x55)  # 0101_0101 pattern
    
    # Set Q8_K d = 1.0 (float32)
    var q8_d = q8_data.unsafe_bitcast[Scalar[DType.float32]]()
    q8_d.unsafe_store(0, 1.0)
    
    # Set Q8_K qs to 1
    for i in range(256):
        q8_data.unsafe_offset(4 + i).unsafe_store(0, 1)
    
    # Set Q8_K bsums
    for i in range(16):
        var bsum = q8_data.unsafe_offset(260 + i * 2).unsafe_bitcast[Scalar[DType.int16]]()
        bsum.unsafe_store(0, 16)
    
    # Run the kernel
    var result = vec_dot_q5_k_q8_k(q5_block, q8_data)
    
    # With q5 = 5 (low bits) and q8 = 1, each element contributes 5
    # 256 elements, each with scale 1
    # Expected: 256 * 5 = 1280
    print("Result: ", result)
    print("Expected: ~1280.0")