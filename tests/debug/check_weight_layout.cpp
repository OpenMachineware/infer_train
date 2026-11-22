// Check weight layout for output projection
#include <cstdio>
#include <cstdint>
#include <cstring>

// Q6_K block size
struct BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

int main() {
    printf("BlockQ6_K size: %zu bytes\n", sizeof(BlockQ6_K));
    printf("Expected: 210 bytes\n");

    // For dims [1024, 151936]:
    // - dim[0] = 1024 (hidden_size)
    // - dim[1] = 151936 (vocab_size)
    // - stride[0] = 210 (block size)
    // - stride[1] = 840 (col_stride)

    // In column-major:
    // Each column (token embedding) is 4 blocks (4 * 256 = 1024 elements)
    // Token j embedding starts at offset j * 840

    int hidden_size = 1024;
    int vocab_size = 151936;
    int blocks_per_token = hidden_size / 256; // = 4

    // For output projection: logits[j] = dot(embedding[:, j], hidden)
    // In column-major, embedding[:, j] is stored contiguously starting at j * 840

    printf("\nWeight layout:\n");
    printf("  hidden_size = %d\n", hidden_size);
    printf("  vocab_size = %d\n", vocab_size);
    printf("  blocks_per_token = %d\n", blocks_per_token);
    printf("  col_stride = %d bytes\n", blocks_per_token * 210);

    printf("\nExpected memory layout:\n");
    printf("  Token 0 embedding: bytes [0, 840)\n");
    printf("  Token 1 embedding: bytes [840, 1680)\n");
    printf("  Token 9707 embedding: bytes [%d, %d)\n", 9707 * 840, 9707 * 840 + 840);

    return 0;
}
