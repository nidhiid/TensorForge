# TensorForge

TensorForge is a focused MLIR-based compiler prototype for transformer inference.
Its first milestone compiles and runs the feed-forward (MLP) portion of a transformer block:

```text
input -> Linear(W1, b1) -> GELU -> Linear(W2, b2) -> output
```

The initial target is static-shape FP32 inference on NVIDIA GPUs. TensorForge will import a small PyTorch graph, represent it in a custom MLIR dialect, optimize it, lower host-side execution to LLVM IR, and dispatch custom CUDA kernels. The first performance optimization is fusing the first Linear and GELU operations so their intermediate tensor does not make a round trip through GPU global memory.

## Project status

Phase 1 froze the initial scope:

- [Project scope](docs/project-scope.md)
- [Acceptance criteria](docs/acceptance-criteria.md)
- [Canonical benchmark workloads](config/workloads.json)

Phase 2 added the first buildable compiler component: a C++17/CMake project and
`tensorforge-opt`, a minimal MLIR command-line tool that parses and prints the
standard `func` dialect.

Phase 3 adds a TableGen-defined TensorForge dialect with verified `tf.matmul`,
`tf.linear`, and `tf.gelu` operations for static FP32 tensors. See the
[dialect reference](docs/dialect.md) for syntax and constraints. Optimization
passes and lowering are not implemented yet.

Phase 4 implements automatic [shape inference](docs/shape-inference.md) using
MLIR's standard type-inference interface. TensorForge operations now derive
their result tensor types directly from their operand shapes.

Phase 5 adds [compiler optimizations](docs/optimizations.md): redundant reshape
and double-transpose canonicalization, constant-reshape folding, and a safe
pass that rewrites a single-use `tf.linear -> tf.gelu` pair to
`tf.fused_linear_gelu`.

Phase 6 adds [CPU lowering and execution](docs/cpu-lowering.md). Every `tf`
operation lowers to standard MLIR `linalg`, `tensor`, `arith`, and `math`
operations. MLIR's existing passes then lower those operations to LLVM IR, and
an end-to-end test JIT-executes fused Linear+GELU and checks its numerical
result.

Phase 7 adds the [PyTorch frontend](docs/pytorch-frontend.md). It captures a
static FP32 `Linear -> GELU(tanh) -> Linear` model with `torch.fx`, converts
PyTorch's weight layout, emits TensorForge MLIR, executes the CPU pipeline, and
checks the complete output tensor against PyTorch.

Phase 8 adds the first [CUDA runtime and MatMul kernel](docs/cuda-runtime.md).
The runtime exposes a stable C API for one device and stream, device memory,
asynchronous copies, synchronization, and row-major FP32 MatMul. The CUDA
kernel uses bounds-checked 16x16 shared-memory tiles, including dimensions that
are not tile multiples. Non-CUDA builds provide the same API as an explicit
"unavailable" stub.

Phase 9 adds [CUDA runtime lowering](docs/cuda-lowering.md). The compiler turns
`tf.matmul`, `tf.linear`, and `tf.fused_linear_gelu` into calls to the runtime,
then MLIR lowers the surrounding host program to LLVM IR. Reshape and transpose
continue through standard MLIR dialects.

Phase 10 adds the [fused CUDA Linear+GELU kernel and evaluation
harness](docs/cuda-linear-gelu.md). The fused kernel performs MatMul, bias, and
the tanh GELU approximation before its single output write. GPU-only tests
compare fused and unfused results, while the Python benchmark uses CUDA events
over all four configured transformer workloads and can save a JSON report.

See [Building TensorForge](docs/building.md) for dependency, build, test, and
execution instructions. CUDA results must be measured on the target NVIDIA GPU;
the repository does not contain invented speedup or hardware-counter numbers.
