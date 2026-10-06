// Dump IQ3 grid data (kmap and neighbors) for Rust quantization
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define QK_K 256

void ggml_quantize_init(int type);
void iq3xs_init_impl(int grid_size);

extern int *iq3_data_map_256;
extern int *iq3_data_map_512;
extern uint16_t *iq3_data_neighbors_256;
extern uint16_t *iq3_data_neighbors_512;

int main() {
    printf("=== IQ3 Grid Data Dump ===\n\n");

    // Initialize IQ3 grid data
    printf("Initializing grid_size=256...\n");
    iq3xs_init_impl(256);

    printf("Initializing grid_size=512...\n");
    iq3xs_init_impl(512);

    printf("\nDone. Data initialized in llama.cpp memory.\n");
    printf("To access: use iq3_data[0].map and iq3_data[0].neighbours for grid_size=256\n");
    printf("           use iq3_data[1].map and iq3_data[1].neighbours for grid_size=512\n");

    return 0;
}
