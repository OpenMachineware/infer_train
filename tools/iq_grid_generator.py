#!/usr/bin/env python3
"""
IQ Grid Table Generator

Generates grid lookup tables for IQ quantization formats using k-means clustering
or product quantization. These grids are used for both quantization and dequantization.

The grid tables in llama.cpp are generated via optimization algorithms, NOT fixed
mathematical formulas. This tool reproduces that process.

Usage:
    python iq_grid_generator.py --format iq3_xxs --output grid_iq3_xxs.json
    python iq_grid_generator.py --format iq2_xxs --samples 100000 --iterations 100
"""

import argparse
import json
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Configuration for a grid table."""
    name: str
    grid_size: int          # Number of entries in grid
    values_per_entry: int   # Number of values per grid entry
    bits_per_value: int     # Bits per value (2, 3, or 4)
    value_range: Tuple[int, int]  # Range of values (e.g., (0x08, 0x2b) for IQ2)

    # For IQ formats, the grid_map tells how to decode packed bytes
    # Example: IQ3_XXS uses grid_map = (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e)


# Standard IQ format configurations
IQ_FORMATS = {
    'iq1_s': GridConfig(
        name='iq1_s',
        grid_size=2048,
        values_per_entry=8,
        bits_per_value=1,
        value_range=(0, 1),  # Binary values
    ),
    'iq2_xxs': GridConfig(
        name='iq2_xxs',
        grid_size=256,
        values_per_entry=8,
        bits_per_value=2,
        value_range=(0x08, 0x19, 0x2b),  # Three possible values
    ),
    'iq2_xs': GridConfig(
        name='iq2_xs',
        grid_size=512,
        values_per_entry=8,
        bits_per_value=2,
        value_range=(0x08, 0x19, 0x2b),
    ),
    'iq2_s': GridConfig(
        name='iq2_s',
        grid_size=1024,
        values_per_entry=8,
        bits_per_value=2,
        value_range=(0x08, 0x19, 0x2b),
    ),
    'iq3_xxs': GridConfig(
        name='iq3_xxs',
        grid_size=256,
        values_per_entry=4,  # Packed as 4 values per entry
        bits_per_value=3,
        value_range=(0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e),  # 8 values
    ),
    'iq3_s': GridConfig(
        name='iq3_s',
        grid_size=512,
        values_per_entry=4,
        bits_per_value=3,
        value_range=(0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f),
    ),
    'iq4_nl': GridConfig(
        name='iq4_nl',
        grid_size=16,
        values_per_entry=1,
        bits_per_value=4,
        value_range=tuple(range(-8, 8)),  # -8 to 7
    ),
    'iq4_xs': GridConfig(
        name='iq4_xs',
        grid_size=256,
        values_per_entry=1,
        bits_per_value=4,
        value_range=tuple(range(16)),  # 0 to 15
    ),
}


def kmeans_quantization(
    data: np.ndarray,
    n_clusters: int,
    n_iter: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform k-means clustering to find optimal quantization grid.

    Args:
        data: Input data, shape (n_samples, n_features)
        n_clusters: Number of cluster centers (grid size)
        n_iter: Number of iterations
        seed: Random seed

    Returns:
        centers: Cluster centers (grid entries)
        labels: Cluster assignments for each sample
    """
    np.random.seed(seed)
    n_samples, n_features = data.shape

    # Initialize centers randomly from data
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centers = data[indices].copy()

    for iteration in range(n_iter):
        # Assign each sample to nearest center
        distances = np.sum((data[:, np.newaxis] - centers) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centers
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = data[mask].mean(axis=0)
            else:
                # Reinitialize empty cluster
                new_centers[k] = data[np.random.randint(n_samples)]

        # Check convergence
        if np.allclose(centers, new_centers):
            print(f"  K-means converged at iteration {iteration}")
            break

        centers = new_centers

    return centers, labels


def product_quantization(
    data: np.ndarray,
    n_subspaces: int,
    n_centroids_per_subspace: int,
    n_iter: int = 100,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Perform product quantization.

    Args:
        data: Input data, shape (n_samples, n_features)
        n_subspaces: Number of subspaces
        n_centroids_per_subspace: Number of centroids per subspace
        n_iter: Number of iterations
        seed: Random seed

    Returns:
        codebooks: List of codebooks for each subspace
    """
    n_samples, n_features = data.shape
    subspace_dim = n_features // n_subspaces

    codebooks = []
    for i in range(n_subspaces):
        start = i * subspace_dim
        end = (i + 1) * subspace_dim
        subspace_data = data[:, start:end]

        centers, _ = kmeans_quantization(
            subspace_data,
            n_centroids_per_subspace,
            n_iter,
            seed + i
        )
        codebooks.append(centers)

    return codebooks


def generate_iq_grid(
    config: GridConfig,
    n_samples: int = 100000,
    n_iter: int = 100,
    seed: int = 42,
    use_importance: bool = False,
    importance_data: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate an IQ grid table using k-means clustering.

    Args:
        config: Grid configuration
        n_samples: Number of random samples for clustering
        n_iter: Number of k-means iterations
        seed: Random seed
        use_importance: Whether to use importance-weighted sampling
        importance_data: Importance weights for sampling (if use_importance=True)

    Returns:
        grid: Grid table as numpy array
    """
    print(f"Generating grid for {config.name}...")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Values per entry: {config.values_per_entry}")
    print(f"  Samples: {n_samples}")

    # Generate random training data
    np.random.seed(seed)

    if use_importance and importance_data is not None:
        # Sample from importance-weighted distribution
        indices = np.random.choice(
            len(importance_data),
            n_samples,
            p=importance_data / importance_data.sum(),
            replace=True
        )
        samples = importance_data[indices]
    else:
        # Generate random samples from typical distribution
        # For neural network activations, typically normal distribution
        samples = np.random.randn(n_samples, config.values_per_entry).astype(np.float32)

    # Quantize each sample to nearest grid value
    value_range = np.array(config.value_range, dtype=np.float32)

    # For product quantization (each dimension quantized independently)
    # We can use k-means on the full vectors

    # Perform k-means clustering
    centers, labels = kmeans_quantization(
        samples,
        config.grid_size,
        n_iter,
        seed
    )

    # Sort centers for consistent ordering
    # Sort by first value, then second, etc.
    sort_indices = np.lexsort([centers[:, i] for i in range(centers.shape[1] - 1, -1, -1)])
    centers = centers[sort_indices]

    # Map to discrete values in value_range
    # This is where we quantize the continuous centers to the discrete grid values
    if len(config.value_range) > 2:
        # Multi-level quantization (like IQ2/IQ3)
        discrete_centers = np.zeros_like(centers)
        for i in range(centers.shape[1]):
            # Find nearest value in range for each dimension
            distances = np.abs(centers[:, i:i+1] - value_range)
            nearest_idx = np.argmin(distances, axis=1)
            discrete_centers[:, i] = value_range[nearest_idx]
    else:
        # Binary or simple range
        discrete_centers = centers

    print(f"  Generated {len(discrete_centers)} grid entries")

    return discrete_centers


def pack_grid_entry(values: np.ndarray, bits_per_value: int) -> int:
    """
    Pack multiple values into a single integer.

    Args:
        values: Array of values to pack
        bits_per_value: Number of bits per value

    Returns:
        Packed integer
    """
    result = 0
    for i, v in enumerate(values):
        result |= (int(v) & ((1 << bits_per_value) - 1)) << (i * bits_per_value)
    return result


def grid_to_hex(grid: np.ndarray, bits_per_value: int) -> bytes:
    """
    Convert grid to hex-encoded bytes (llama.cpp format).

    Args:
        grid: Grid table
        bits_per_value: Number of bits per value

    Returns:
        Hex-encoded bytes
    """
    # Pack each row of the grid
    packed = []
    for row in grid:
        packed.append(pack_grid_entry(row, bits_per_value))

    # Convert to bytes
    # Determine bytes per entry based on total bits
    total_bits = len(grid[0]) * bits_per_value
    if total_bits <= 8:
        dtype = np.uint8
    elif total_bits <= 16:
        dtype = np.uint16
    elif total_bits <= 32:
        dtype = np.uint32
    else:
        dtype = np.uint64

    packed_array = np.array(packed, dtype=dtype)
    return packed_array.tobytes()


def export_grid_json(grid: np.ndarray, config: GridConfig, output_path: str):
    """Export grid as JSON file."""
    data = {
        'name': config.name,
        'grid_size': config.grid_size,
        'values_per_entry': config.values_per_entry,
        'bits_per_value': config.bits_per_value,
        'value_range': list(config.value_range),
        'grid': grid.tolist(),
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {output_path}")


def export_grid_c_header(grid: np.ndarray, config: GridConfig, output_path: str):
    """Export grid as C header file (llama.cpp format)."""
    with open(output_path, 'w') as f:
        f.write(f"// Auto-generated grid table for {config.name}\n")
        f.write(f"// Grid size: {config.grid_size}, Values per entry: {config.values_per_entry}\n\n")

        total_bits = config.values_per_entry * config.bits_per_value
        if total_bits <= 32:
            dtype = "uint32_t"
        else:
            dtype = "uint64_t"

        f.write(f"GGML_TABLE_BEGIN({dtype}, {config.name}_grid, {config.grid_size})\n")

        for i, row in enumerate(grid):
            packed = pack_grid_entry(row, config.bits_per_value)
            if i % 4 == 0:
                f.write("    ")
            f.write(f"0x{packed:0{(total_bits + 3) // 4}x}, ")
            if (i + 1) % 4 == 0:
                f.write("\n")

        if len(grid) % 4 != 0:
            f.write("\n")

        f.write("GGML_TABLE_END()\n")

    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate IQ grid tables')
    parser.add_argument('--format', '-f', required=True,
                        choices=list(IQ_FORMATS.keys()),
                        help='IQ format name')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path (default: grid_<format>.json)')
    parser.add_argument('--samples', '-s', type=int, default=100000,
                        help='Number of random samples for clustering')
    parser.add_argument('--iterations', '-i', type=int, default=100,
                        help='Number of k-means iterations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--format-output', choices=['json', 'c_header', 'both'],
                        default='json', help='Output format')

    args = parser.parse_args()

    config = IQ_FORMATS[args.format]
    grid = generate_iq_grid(
        config,
        n_samples=args.samples,
        n_iter=args.iterations,
        seed=args.seed
    )

    output_path = args.output or f"grid_{args.format}.json"

    if args.format_output in ('json', 'both'):
        export_grid_json(grid, config, output_path)

    if args.format_output in ('c_header', 'both'):
        header_path = output_path.replace('.json', '.h')
        export_grid_c_header(grid, config, header_path)

    print("\nGrid generation complete!")


if __name__ == '__main__':
    main()
