# InferTrain TODO List

Welcome to contribute! The following tasks are sorted by priority.


## I. Operator Completion (P0 - Highest Priority)

The current operator library has ~78 operators, with a goal of 400+. Below is the list of missing operators.

### 1.1 Math Operators

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `erf` | Error function | ⬜ To be implemented | ⭐ |
| `erfc` | Complementary error function | ⬜ To be implemented | ⭐ |
| `sinc` | Sinc function | ⬜ To be implemented | ⭐ |
| `sign` | Sign function | ⬜ To be implemented | ⭐ |
| `rsqrt` | Reciprocal square root | ⬜ To be implemented | ⭐ |
| `log1p` | log(1+x) | ⬜ To be implemented | ⭐ |
| `expm1` | exp(x)-1 | ⬜ To be implemented | ⭐ |
| `cbrt` | Cube root | ⬜ To be implemented | ⭐ |
| `rad2deg` | Radians to degrees | ⬜ To be implemented | ⭐ |
| `deg2rad` | Degrees to radians | ⬜ To be implemented | ⭐ |
| `digamma` | Digamma function | ⬜ To be implemented | ⭐⭐ |
| `trigamma` | Trigamma function | ⬜ To be implemented | ⭐⭐ |
| `zeta` | Riemann zeta function | ⬜ To be implemented | ⭐⭐⭐ |

### 1.2 Trigonometric/Hyperbolic Functions

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `asin` | Arcsine | ⬜ To be implemented | ⭐ |
| `acos` | Arccosine | ⬜ To be implemented | ⭐ |
| `atan` | Arctangent | ⬜ To be implemented | ⭐ |
| `asinh` | Inverse hyperbolic sine | ⬜ To be implemented | ⭐ |
| `acosh` | Inverse hyperbolic cosine | ⬜ To be implemented | ⭐ |
| `atanh` | Inverse hyperbolic tangent | ⬜ To be implemented | ⭐ |
| `sinh` | Hyperbolic sine | ⬜ To be implemented | ⭐ |
| `cosh` | Hyperbolic cosine | ⬜ To be implemented | ⭐ |
| `atan2` | Two-argument arctangent | ⬜ To be implemented | ⭐⭐ |

### 1.3 Linear Algebra (Linalg)

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `batch_matmul` | Batch matrix multiplication | ⬜ To be implemented | ⭐⭐⭐ |
| `matmul_transpose_a` | A^T @ B | ⬜ To be implemented | ⭐⭐ |
| `matmul_transpose_b` | A @ B^T | ⬜ To be implemented | ⭐⭐ |
| `matmul_transpose_ab` | A^T @ B^T | ⬜ To be implemented | ⭐⭐ |
| `trace` | Matrix trace | ⬜ To be implemented | ⭐ |
| `det` | Determinant | ⬜ To be implemented | ⭐⭐ |
| `inverse` | Matrix inverse | ⬜ To be implemented | ⭐⭐⭐ |
| `eig` | Eigenvalues/eigenvectors | ⬜ To be implemented | ⭐⭐⭐ |
| `svd` | Singular value decomposition | ⬜ To be implemented | ⭐⭐⭐ |
| `qr` | QR decomposition | ⬜ To be implemented | ⭐⭐⭐ |
| `lu` | LU decomposition | ⬜ To be implemented | ⭐⭐⭐ |
| `cholesky` | Cholesky decomposition | ⬜ To be implemented | ⭐⭐⭐ |
| `norm_l1` | L1 norm | ⬜ To be implemented | ⭐ |
| `norm_l2` | L2 norm | ⬜ To be implemented | ⭐ |
| `norm_inf` | Infinity norm | ⬜ To be implemented | ⭐ |
| `norm_fro` | Frobenius norm | ⬜ To be implemented | ⭐ |

### 1.4 Activation Functions

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `hard_swish` | Hard Swish | ⬜ To be implemented | ⭐ |
| `hard_sigmoid` | Hard Sigmoid | ⬜ To be implemented | ⭐ |
| `softsign` | Softsign | ⬜ To be implemented | ⭐ |
| `softshrink` | Softshrink | ⬜ To be implemented | ⭐ |
| `tanhshrink` | Tanhshrink | ⬜ To be implemented | ⭐ |
| `threshold` | Threshold activation | ⬜ To be implemented | ⭐ |
| `rrelu` | Randomized Leaky ReLU | ⬜ To be implemented | ⭐⭐ |
| `prelu` | Parametric ReLU | ⬜ To be implemented | ⭐⭐ |
| `celu` | CELU | ⬜ To be implemented | ⭐⭐ |
| `selu` | SELU | ⬜ To be implemented | ⭐⭐ |
| `mish` | Mish | ⬜ To be implemented | ⭐⭐ |
| `swish` | Swish (SiLU alias) | ⬜ To be implemented | ⭐ |
| `hardshrink` | Hard Shrink | ⬜ To be implemented | ⭐ |

### 1.5 Convolution & Pooling

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `conv1d` | 1D convolution | ⬜ To be implemented | ⭐⭐⭐ |
| `conv3d` | 3D convolution | ⬜ To be implemented | ⭐⭐⭐ |
| `conv_transpose1d` | 1D transposed convolution | ⬜ To be implemented | ⭐⭐⭐ |
| `conv_transpose2d` | 2D transposed convolution | ⬜ To be implemented | ⭐⭐⭐ |
| `conv_transpose3d` | 3D transposed convolution | ⬜ To be implemented | ⭐⭐⭐ |
| `dilated_conv2d` | Dilated convolution | ⬜ To be implemented | ⭐⭐ |
| `depthwise_conv2d` | Depthwise separable convolution | ⬜ To be implemented | ⭐⭐ |
| `group_conv2d` | Grouped convolution | ⬜ To be implemented | ⭐⭐ |
| `maxpool1d` | 1D max pooling | ⬜ To be implemented | ⭐⭐ |
| `maxpool3d` | 3D max pooling | ⬜ To be implemented | ⭐⭐ |
| `avgpool1d` | 1D average pooling | ⬜ To be implemented | ⭐⭐ |
| `avgpool3d` | 3D average pooling | ⬜ To be implemented | ⭐⭐ |
| `adaptive_maxpool2d` | Adaptive max pooling | ⬜ To be implemented | ⭐⭐ |
| `adaptive_avgpool2d` | Adaptive average pooling | ⬜ To be implemented | ⭐⭐ |
| `lp_pool2d` | Lp pooling | ⬜ To be implemented | ⭐⭐⭐ |
| `fractional_maxpool2d` | Fractional max pooling | ⬜ To be implemented | ⭐⭐⭐ |
| `unpooling` | Unpooling | ⬜ To be implemented | ⭐⭐ |
| `upsample_nearest` | Nearest neighbor upsampling | ⬜ To be implemented | ⭐ |
| `upsample_bilinear` | Bilinear upsampling | ⬜ To be implemented | ⭐⭐ |
| `upsample_trilinear` | Trilinear upsampling | ⬜ To be implemented | ⭐⭐⭐ |

### 1.6 Normalization

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `batch_norm1d` | 1D batch normalization | ⬜ To be implemented | ⭐⭐⭐ |
| `batch_norm3d` | 3D batch normalization | ⬜ To be implemented | ⭐⭐⭐ |
| `layer_norm` | Layer normalization | ⬜ To be implemented | ⭐⭐ |
| `rms_norm` | RMS normalization | ⬜ To be implemented | ⭐⭐ |
| `instance_norm1d/2d/3d` | Instance normalization | ⬜ To be implemented | ⭐⭐ |
| `group_norm` | Group normalization | ⬜ To be implemented | ⭐⭐⭐ |
| `local_response_norm` | Local response normalization | ⬜ To be implemented | ⭐⭐ |
| `weight_norm` | Weight normalization | ⬜ To be implemented | ⭐⭐ |
| `spectral_norm` | Spectral normalization | ⬜ To be implemented | ⭐⭐⭐ |

### 1.7 Loss Functions

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `nll_loss` | Negative log-likelihood loss | ⬜ To be implemented | ⭐⭐ |
| `kl_div` | KL divergence | ⬜ To be implemented | ⭐⭐ |
| `huber_loss` | Huber loss | ⬜ To be implemented | ⭐⭐ |
| `smooth_l1_loss` | Smooth L1 loss | ⬜ To be implemented | ⭐⭐ |
| `hinge_loss` | Hinge loss | ⬜ To be implemented | ⭐ |
| `multi_margin_loss` | Multi-class margin loss | ⬜ To be implemented | ⭐⭐ |
| `triplet_margin_loss` | Triplet margin loss | ⬜ To be implemented | ⭐⭐⭐ |
| `cosine_embedding_loss` | Cosine embedding loss | ⬜ To be implemented | ⭐⭐ |
| `ctc_loss` | CTC loss | ⬜ To be implemented | ⭐⭐⭐ |

### 1.8 Reduction

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `any` | Logical OR reduction | ⬜ To be implemented | ⭐ |
| `all` | Logical AND reduction | ⬜ To be implemented | ⭐ |
| `std` | Standard deviation | ⬜ To be implemented | ⭐⭐ |
| `var` | Variance | ⬜ To be implemented | ⭐⭐ |
| `cov` | Covariance | ⬜ To be implemented | ⭐⭐⭐ |
| `corrcoef` | Correlation coefficient | ⬜ To be implemented | ⭐⭐⭐ |
| `median` | Median | ⬜ To be implemented | ⭐⭐ |
| `quantile` | Quantile | ⬜ To be implemented | ⭐⭐ |

### 1.9 Tensor Manipulation

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `split` | Split | ⬜ To be implemented | ⭐⭐ |
| `chunk` | Chunk | ⬜ To be implemented | ⭐⭐ |
| `stack` | Stack | ⬜ To be implemented | ⭐⭐ |
| `unbind` | Unbind | ⬜ To be implemented | ⭐⭐ |
| `repeat` | Repeat | ⬜ To be implemented | ⭐⭐ |
| `tile` | Tile | ⬜ To be implemented | ⭐⭐ |
| `roll` | Roll | ⬜ To be implemented | ⭐⭐ |
| `rot90` | Rotate 90 degrees | ⬜ To be implemented | ⭐⭐ |
| `flip` | Flip | ⬜ To be implemented | ⭐ |
| `gather` | Gather | ⬜ To be implemented | ⭐⭐ |
| `scatter` | Scatter | ⬜ To be implemented | ⭐⭐ |
| `index_select` | Index select | ⬜ To be implemented | ⭐ |
| `masked_select` | Masked select | ⬜ To be implemented | ⭐⭐ |
| `nonzero` | Non-zero elements | ⬜ To be implemented | ⭐⭐ |
| `where` | Where | ⬜ To be implemented | ⭐⭐ |
| `take` | Take | ⬜ To be implemented | ⭐ |
| `put` | Put | ⬜ To be implemented | ⭐ |

### 1.10 Embedding & Lookup

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `embedding` | Embedding | ⬜ To be implemented | ⭐⭐ |
| `one_hot` | One-hot encoding | ⬜ To be implemented | ⭐ |
| `topk` | Top-K | ⬜ To be implemented | ⭐⭐ |
| `sort` | Sort | ⬜ To be implemented | ⭐⭐ |
| `unique` | Unique | ⬜ To be implemented | ⭐⭐ |

### 1.11 Attention

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `scaled_dot_product_attention` | Scaled dot-product attention | ⬜ To be implemented | ⭐⭐⭐ |
| `multi_head_attention` | Multi-head attention | ⬜ To be implemented | ⭐⭐⭐ |
| `flash_attention` | Flash Attention | ⬜ To be implemented | ⭐⭐⭐⭐ |
| `rotary_embedding` | Rotary positional embedding | ⬜ To be implemented | ⭐⭐ |
| `alibi` | Alibi positional encoding | ⬜ To be implemented | ⭐⭐ |
| `kv_cache` | KV Cache management | ⬜ To be implemented | ⭐⭐⭐ |

### 1.12 Control Flow

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `if` | Conditional branch | ⬜ To be implemented | ⭐⭐⭐ |
| `while` | While loop | ⬜ To be implemented | ⭐⭐⭐ |
| `for` | For loop | ⬜ To be implemented | ⭐⭐⭐ |
| `switch` | Switch | ⬜ To be implemented | ⭐⭐⭐ |
| `cond` | Conditional execution | ⬜ To be implemented | ⭐⭐ |

### 1.13 Quantized Operators

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `quantized_matmul` | Quantized matrix multiplication | ⬜ To be implemented | ⭐⭐⭐ |
| `quantized_conv2d` | Quantized convolution | ⬜ To be implemented | ⭐⭐⭐ |
| `quantized_linear` | Quantized linear layer | ⬜ To be implemented | ⭐⭐⭐ |
| `quantized_lstm` | Quantized LSTM | ⬜ To be implemented | ⭐⭐⭐ |
| `quantized_gru` | Quantized GRU | ⬜ To be implemented | ⭐⭐⭐ |
| `quantized_add` | Quantized addition | ⬜ To be implemented | ⭐ |
| `quantized_mul` | Quantized multiplication | ⬜ To be implemented | ⭐ |
| `quantized_relu` | Quantized ReLU | ⬜ To be implemented | ⭐ |
| `quantized_sigmoid` | Quantized Sigmoid | ⬜ To be implemented | ⭐ |
| `quantized_tanh` | Quantized Tanh | ⬜ To be implemented | ⭐ |
| `dequantize` | Dequantize | ⬜ To be implemented | ⭐ |
| `quantize` | Quantize | ⬜ To be implemented | ⭐ |

### 1.14 Sparse Operators

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `sparse_matmul` | Sparse matrix multiplication | ⬜ To be implemented | ⭐⭐⭐ |
| `sparse_conv2d` | Sparse convolution | ⬜ To be implemented | ⭐⭐⭐ |
| `sparse_to_dense` | Sparse to dense | ⬜ To be implemented | ⭐⭐ |
| `dense_to_sparse` | Dense to sparse | ⬜ To be implemented | ⭐⭐ |
| `sparse_embedding` | Sparse embedding | ⬜ To be implemented | ⭐⭐ |

### 1.15 Data Generation

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `rand` | Uniform distribution | ⬜ To be implemented | ⭐ |
| `randn` | Normal distribution | ⬜ To be implemented | ⭐ |
| `randint` | Random integer | ⬜ To be implemented | ⭐ |
| `normal` | Normal distribution (parametric) | ⬜ To be implemented | ⭐ |
| `uniform` | Uniform distribution (parametric) | ⬜ To be implemented | ⭐ |
| `bernoulli` | Bernoulli distribution | ⬜ To be implemented | ⭐ |
| `multinomial` | Multinomial distribution | ⬜ To be implemented | ⭐⭐ |

### 1.16 Recurrent Networks (RNN)

| Operator | Description | Status | Difficulty |
|------|------|------|------|
| `lstm` | LSTM | ⬜ To be implemented | ⭐⭐⭐ |
| `gru` | GRU | ⬜ To be implemented | ⭐⭐⭐ |
| `rnn` | RNN | ⬜ To be implemented | ⭐⭐⭐ |
| `lstm_cell` | LSTM cell | ⬜ To be implemented | ⭐⭐ |
| `gru_cell` | GRU cell | ⬜ To be implemented | ⭐⭐ |


## II. Automatic Differentiation (Autograd) (P0)

| Task | Description | Status | Difficulty |
|------|------|------|------|
| Full Backward Coverage | All operators need backward | ⬜ To be implemented | ⭐⭐⭐ |
| Gradient Checkpointing | Memory optimization, compute for memory | ⬜ To be implemented | ⭐⭐⭐ |
| Mixed Precision Training | FP16/BF16 automatic mixed precision | ⬜ To be implemented | ⭐⭐⭐ |
| Gradient Clipping | Prevent gradient explosion | ⬜ To be implemented | ⭐ |
| Gradient Accumulation | Multi-batch gradient accumulation | ⬜ To be implemented | ⭐⭐ |
| Gradient Check | Numeric gradient verification | ⬜ To be implemented | ⭐⭐ |
| Second-order Gradient (Hessian) | Second-order optimization support | ⬜ To be implemented | ⭐⭐⭐⭐ |
| Autograd Documentation | Complete API documentation | ⬜ To be implemented | ⭐ |


## III. Optimization Passes (P0)

| Task | Description | Status | Difficulty |
|------|------|-------|------|
| Constant Folding | Compile-time constant computation | ✅ Implemented | ⭐⭐ |
| Common Subexpression Elimination (CSE) | Eliminate redundant computations | ✅ Implemented | ⭐⭐⭐ |
| Dead Code Elimination (DCE) | Remove unused code | ✅ Implemented | ⭐⭐ |
| Operator Fusion | Conv+BN+ReLU etc. | ✅ Implemented | ⭐⭐⭐ |
| Algebraic Simplification | x*1→x, x+0→x | ✅ Implemented | ⭐⭐ |
| Shape Inference | Automatic tensor shape derivation | ✅ Implemented | ⭐⭐⭐ |
| Memory Planning | Optimize memory allocation | ✅ Implemented | ⭐⭐⭐ |
| Operation Reordering | Optimize execution order | ✅ Implemented | ⭐⭐⭐ |
| Autograd Fusion | Forward+backward fusion optimization | ⬜ To be implemented | ⭐⭐⭐⭐ |
| Graph Folding | Merge multiple small operators | ⬜ To be implemented | ⭐⭐⭐ |
| Dimension Inference | Automatic dynamic dimension inference | ⬜ To be implemented | ⭐⭐ |


## IV. Hardware Acceleration (P1)

### 4.1 Instruction Set Optimization

| Task | Description | Status | Difficulty |
|------|------|------|------|
| AVX2 Support | Intel/AMD SIMD acceleration | ⬜ To be implemented | ⭐⭐⭐ |
| AVX-512 Support | Advanced Vector Extensions | ⬜ To be implemented | ⭐⭐⭐ |
| NEON Support | ARM SIMD acceleration | ⬜ To be implemented | ⭐⭐⭐ |
| SVE Support | ARM Scalable Vector Extensions | ⬜ To be implemented | ⭐⭐⭐ |
| Auto-vectorization | Compiler automatic vectorization | ⬜ To be implemented | ⭐⭐ |
| Loop Unrolling | Manual loop unrolling optimization | ⬜ To be implemented | ⭐⭐ |
| Memory Alignment | Data alignment optimization | ⬜ To be implemented | ⭐⭐ |
| Cache Prefetch | Prefetch instruction optimization | ⬜ To be implemented | ⭐⭐ |

### 4.2 Apple Platform

| Task | Description | Status | Difficulty |
|------|------|------|------|
| Apple GPU (Metal) | Integrate Metal acceleration | ⬜ To be implemented | ⭐⭐⭐⭐ |
| Apple GPU Operators | Metal operator library | ⬜ To be implemented | ⭐⭐⭐ |
| Apple Memory Management | Unified memory architecture | ⬜ To be implemented | ⭐⭐⭐ |
| Metal Performance Shaders | MPS integration | ⬜ To be implemented | ⭐⭐⭐ |

### 4.3 NVIDIA Platform

| Task | Description | Status | Difficulty |
|------|------|------|------|
| CUDA Support | Integrate CUDA acceleration | ⬜ To be implemented | ⭐⭐⭐⭐ |
| cuBLAS Integration | Matrix operation acceleration | ⬜ To be implemented | ⭐⭐⭐ |
| cuDNN Integration | Deep neural network acceleration | ⬜ To be implemented | ⭐⭐⭐⭐ |
| Tensor Cores | Tensor core acceleration | ⬜ To be implemented | ⭐⭐⭐ |
| Multi-GPU Support | Distributed training/inference | ⬜ To be implemented | ⭐⭐⭐⭐⭐ |

### 4.4 AMD Platform

| Task | Description | Status | Difficulty |
|------|------|------|------|
| ROCm Support | Integrate ROCm acceleration | ⬜ To be implemented | ⭐⭐⭐⭐ |
| rocBLAS Integration | Matrix operation acceleration | ⬜ To be implemented | ⭐⭐⭐ |
| MIOpen Integration | Deep neural network acceleration | ⬜ To be implemented | ⭐⭐⭐⭐ |


## V. Frontend Support (P1)

| Task | Description | Status | Difficulty |
|------|------|------|------|
| ONNX Import | ONNX model parsing | ⬜ To be implemented | ⭐⭐⭐ |
| TensorFlow Import | TensorFlow SavedModel | ⬜ To be implemented | ⭐⭐⭐ |
| JAX Import | JAX model import | ⬜ To be implemented | ⭐⭐⭐ |
| PyTorch Export | Export to PyTorch | ⬜ To be implemented | ⭐⭐ |
| TensorFlow Lite Export | TFLite support | ⬜ To be implemented | ⭐⭐⭐ |


## VI. Storage Format (P1)

| Task | Description | Status | Difficulty |
|------|------|------|------|
| ITM Format Optimization | Chunked storage, mmap support | ⬜ To be implemented | ⭐⭐ |
| Weight Compression | Lossless/lossy compression | ⬜ To be implemented | ⭐⭐ |
| Incremental Save | Save only changed parts | ⬜ To be implemented | ⭐⭐ |
| Model Sharding | Large model sharding | ⬜ To be implemented | ⭐⭐⭐ |
| Encryption Support | Model encryption storage | ⬜ To be implemented | ⭐⭐⭐ |
| Checkpoint | Training resume | ⬜ To be implemented | ⭐⭐ |
| Version Compatibility | Forward/backward compatibility | ⬜ To be implemented | ⭐⭐ |


## VII. Quantization & Compression (P2)

| Task | Description | Status | Difficulty |
|------|------|------|------|
| Quantization-Aware Training (QAT) | Simulate quantization during training | ⬜ To be implemented | ⭐⭐⭐ |
| Post-Training Quantization (PTQ) | Quantization after training | ⬜ To be implemented | ⭐⭐⭐ |
| INT4 Support | 4-bit quantization | ⬜ To be implemented | ⭐⭐ |
| INT2 Support | 2-bit quantization | ⬜ To be implemented | ⭐⭐⭐ |
| Mixed Precision Quantization | Different precision per layer | ⬜ To be implemented | ⭐⭐⭐ |
| Sparsification | Weight sparsification | ⬜ To be implemented | ⭐⭐⭐ |
| Pruning | Structured/unstructured pruning | ⬜ To be implemented | ⭐⭐⭐ |
| Knowledge Distillation | Distillation training support | ⬜ To be implemented | ⭐⭐⭐⭐ |
| Weight Replay | Incremental training compatibility | ⬜ To be implemented | ⭐⭐⭐ |


## VIII. Performance Optimization (P2)

| Task | Description | Status | Difficulty |
|------|------|--------|------|
| Memory Reuse Pool | Byte-pool memory management | ✅ Implemented | ⭐⭐⭐ |
| Memory Pool | General memory pool | ✅ Implemented | ⭐⭐ |
| Parallel Execution | Rayon parallel | ✅ Implemented | ⭐⭐⭐ |
| Asynchronous Execution | Asynchronous computation scheduling | ⬜ To be implemented | ⭐⭐⭐ |
| Pipeline Execution | Operator pipeline | ⬜ To be implemented | ⭐⭐⭐ |
| Graph Optimization | Multi-round graph optimization | ✅ Implemented | ⭐⭐⭐ |
| JIT Compilation | Runtime compilation optimization | ⬜ To be implemented | ⭐⭐⭐⭐ |
| Fused Kernel | Multi-operator fused kernel | ⬜ To be implemented | ⭐⭐⭐⭐ |
| Zero Copy | Zero-copy data transfer | ⬜ To be implemented | ⭐⭐⭐ |


## IX. Python Bindings (P2)

| Task | Description | Status | Difficulty |
|------|------|------|------|
| PyO3 Binding Optimization | Reduce Python overhead | ⬜ To be implemented | ⭐⭐⭐ |
| NumPy Interop | Direct NumPy array conversion | ⬜ To be implemented | ⭐⭐ |
| PyTorch Interop | Seamless PyTorch Tensor conversion | ⬜ To be implemented | ⭐⭐ |
| Batch Processing Optimization | Python-side batch interface | ⬜ To be implemented | ⭐⭐ |
| JIT Trace Optimization | TorchScript tracing | ⬜ To be implemented | ⭐⭐ |
| Hook Non-Invasive Integration | Complete Hook mechanism | ⬜ To be implemented | ⭐⭐ |


## X. Testing & Validation (P2)

| Task | Description | Status | Difficulty |
|------|------|------|------|
| Unit Test Coverage | 90%+ coverage | ⬜ To be implemented | ⭐⭐ |
| Integration Test | End-to-end testing | ⬜ To be implemented | ⭐⭐ |
| Performance Benchmark | Performance regression testing | ⬜ To be implemented | ⭐⭐ |
| Accuracy Validation | Compare with PyTorch | ⬜ To be implemented | ⭐⭐ |
| Memory Leak Detection | Memory testing | ⬜ To be implemented | ⭐⭐ |
| Quantization Accuracy Test | Quantized model accuracy | ⬜ To be implemented | ⭐⭐ |
| Edge Device Testing | Real device validation | ⬜ To be implemented | ⭐⭐⭐ |


## XI. Documentation & Community (P3)

| Task | Description | Status | Difficulty |
|------|------|------|------|
| API Documentation | Complete API documentation | ⬜ To be implemented | ⭐⭐ |
| Tutorial | Quick start guide | ⬜ To be implemented | ⭐ |
| Examples | Common scenario examples | ⬜ To be implemented | ⭐ |
| Operator Development Guide | New operator addition guide | ✅ Completed | - |
| Hardware Integration Guide | New platform integration guide | ✅ Completed | - |
| Performance Tuning Guide | Best practices | ⬜ To be implemented | ⭐⭐ |
| Chinese Documentation | Complete Chinese translation | ⬜ To be implemented | ⭐⭐ |


## How to Contribute

### Adding New Operators

1. Refer to [docs/dev_ops.md](./docs/dev_ops.md)
2. Add in `src/ops/` corresponding directory
3. Implement Forward + Backward + Quantized
4. Add tests
5. Update `ops.rs`

### Adding Hardware Support

1. Refer to [docs/dev_platform.md](./docs/dev_platform.md)
2. Implement `Operator<T>` trait
3. Implement device memory management
4. Register to engine

### Submitting Code

```bash
# 1. Fork the repository
# 2. Create branch
git checkout -b feature/my-operator

# 3. Commit code
git commit -m -s "add: my operator implementation"

# 4. Create Pull Request
```

### Task Assignment

Please claim tasks in GitHub Issues or specify in PR.


## Current Progress

| Category | Completed | Total | Progress |
|------|--------|------|------|
| Math Operators | 12 | 27 | 44% |
| Linear Algebra | 4 | 19 | 21% |
| Activation Functions | 10 | 22 | 45% |
| Convolution & Pooling | 6 | 28 | 21% |
| Normalization | 2 | 11 | 18% |
| Loss Functions | 2 | 11 | 18% |
| Reduction | 4 | 12 | 33% |
| Tensor Manipulation | 5 | 18 | 28% |
| Embedding & Lookup | 0 | 7 | 0% |
| Attention | 0 | 6 | 0% |
| Control Flow | 0 | 5 | 0% |
| Quantized Operators | 0 | 12 | 0% |
| Sparse Operators | 0 | 5 | 0% |
| Data Generation | 0 | 7 | 0% |
| Recurrent Networks | 0 | 5 | 0% |
| **Total** | **45** | **195** | **23%** |