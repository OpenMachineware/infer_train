#!/bin/bash
set -e

# Remove include directive from mul_mm_standalone.metal
sed 's/#include "iq_grid_tables.h"//' shaders/mul_mm_standalone.metal > /tmp/mul_mm_no_include.metal

# Combine sources
cat shaders/ggml-common.h shaders/iq_grid_tables.h /tmp/mul_mm_no_include.metal > /tmp/mul_mm_combined.metal

# Compile
xcrun -sdk macosx metal -O3 -c /tmp/mul_mm_combined.metal -o shaders/mul_mm_standalone.metallib

echo "Compiled mul_mm_standalone.metallib successfully"
