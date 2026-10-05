// Simple test to dump llama.cpp IQ2_XXS quantization intermediate values
// Run: llama-cli -m test.gguf --no-cpu-mask -p "test" 2>&1 | grep "LLAMA.CPP DUMP"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// We'll use a different approach: create a simple model file and run inference
// But that's complex. Instead, let's just write the test values to a file
// and call the quantization function directly.

// Actually, let's use llama-cli with a tiny model and check the dump output.

int main() {
    printf("To see llama.cpp dump output:\n");
    printf("1. The dump code is now in ggml-quants.c\n");
    printf("2. Run a llama-cli command that triggers IQ2_XXS quantization\n");
    printf("\n");
    printf("Or use the Python test to call the C quantization function directly.\n");
    return 0;
}