#import <Metal/Metal.h>
static const char* llama_kernel = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03; int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13; int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;
typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
kernel void kernel_mul_mv_q5_K_f32(constant ggml_metal_kargs_mul_mv & args [[buffer(0)]], device const char * src0 [[buffer(1)]], device const char * src1 [[buffer(2)]], device char * dst [[buffer(3)]], uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // Stub
}
)METAL";
int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:llama_kernel] options:nil error:&error];
        if (!lib) { printf("Error: %s\n", [[error localizedDescription] UTF8String]); return 1; }
        id<MTLFunction> func = [lib newFunctionWithName:@"kernel_mul_mv_q5_K_f32"];
        printf("Function: %s\n", func ? "found" : "not found");
    }
    return 0;
}
