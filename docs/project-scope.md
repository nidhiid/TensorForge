# TensorForge v0.1 Project Scope

## Objective

Build a small, end-to-end compiler that accepts a supported PyTorch transformer
MLP, applies visible MLIR transformations, and executes it using CUDA kernels.
The project is intended to demonstrate compiler construction and GPU
optimization, not to replace PyTorch or compile a complete transformer.

## Supported graph

Version 0.1 supports this complete graph:

```text
x[B, S, H]
  -> Linear(H, I) + bias[I]
  -> GELU
  -> Linear(I, H) + bias[H]
  -> y[B, S, H]
```

During compilation, `B` and `S` may be flattened to `M = B * S`, producing the
equivalent 2D computation:

```text
[M, H] x [H, I] -> [M, I] -> GELU -> [M, I] x [I, H] -> [M, H]
```

All dimensions are compile-time constants in v0.1.

## Supported operations

The frontend and TensorForge dialect will support only:

| Operation | Required behavior |
| --- | --- |
| `matmul` | Rank-2 matrix multiplication with compatible inner dimensions |
| `linear` | Matrix multiplication followed by a rank-1 bias addition |
| `bias_add` | Broadcast a rank-1 bias across the final tensor dimension |
| `gelu` | Elementwise GELU using one documented approximation consistently |
| `reshape` | Static reshape with an unchanged element count |
| `transpose` | Static rank-2 transpose initially |

The optimized dialect will additionally contain `fused_linear_gelu`, produced
by a compiler pass rather than required as a frontend input operation.

## Input and execution constraints

- Inference only; no gradients or training graph.
- FP32 tensors only in v0.1.
- Dense, contiguous tensors only.
- Static shapes only.
- NVIDIA CUDA GPU target.
- One CUDA stream and one device per process.
- Batch and sequence dimensions are flattened before matrix kernels run.
- Weights and biases are treated as immutable during one compiled execution.
- Linux is the primary supported build environment.

## Compiler boundary

TensorForge is responsible for:

1. Importing the supported operations from a PyTorch FX graph.
2. Validating types, ranks, and dimensions.
3. Inferring result shapes.
4. Applying constant folding and canonicalization.
5. Fusing `linear -> gelu` when it is safe.
6. Selecting a compatible CUDA tiling configuration.
7. Lowering host-side execution and CUDA runtime calls to LLVM IR.
8. Running the compiled graph and returning its result to Python.

The CUDA runtime is responsible for device allocation, transfers, kernel
launches, synchronization, and error reporting. CUDA kernels are responsible
for MatMul and fused Linear+GELU computation.

## Explicitly out of scope for v0.1

- Full transformer or arbitrary PyTorch model support
- Attention, softmax, layer normalization, embeddings, and KV caches
- Dynamic shapes
- Training and automatic differentiation
- FP16, BF16, INT8, quantization, and mixed precision
- AMD, Apple, Intel GPU, or multi-GPU execution
- Distributed execution
- Sparse tensors
- General control flow
- Direct PTX generation
- Production-grade memory planning or kernel autotuning

These may be future extensions, but none is required for declaring v0.1
complete.
## Canonical workloads

The checked-in workload definitions in `config/workloads.json` are the source
of truth. They represent:

| Workload | B | S | H | I |
| --- | ---: | ---: | ---: | ---: |
| BERT Base | 1 | 128 | 768 | 3072 |
| BERT Large | 1 | 128 | 1024 | 4096 |
| GPT-2 Small | 1 | 256 | 768 | 3072 |
| ViT Base | 1 | 197 | 768 | 3072 |

Synthetic random tensors with a fixed seed will be used so benchmarks are
reproducible and do not require downloading model weights.

## Performance claims

Speedup and DRAM-traffic reduction are experimental results, not requirements
or constants. The implementation must measure them before placing numbers in a
README or resume. Compilation time must be reported separately from inference
latency, and every timed GPU region must use explicit CUDA synchronization or
CUDA events.
