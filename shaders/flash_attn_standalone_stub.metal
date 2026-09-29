// Stub Flash Attention for testing
// Real implementation uses template specialization which requires Metal function constants

#include <metal_stdlib>
using namespace metal;

// Empty kernel for compilation
kernel void kernel_flash_attn_stub(device float* dst [[buffer(0)]], uint idx [[thread_position_in_grid]]) {
    if (idx < 1024) {
        dst[idx] = 0.0f;
    }
}
