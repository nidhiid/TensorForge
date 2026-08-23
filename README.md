# TensorForge

TensorForge is a focused MLIR-based compiler prototype for transformer inference.
Its first milestone compiles and runs the feed-forward (MLP) portion of a transformer block:

```text
input -> Linear(W1, b1) -> GELU -> Linear(W2, b2) -> output
```

The initial target is static-shape FP32 inference on NVIDIA GPUs. TensorForge will import a small PyTorch graph, represent it in a custom MLIR dialect, optimize it, lower host-side execution to LLVM IR, and dispatch custom CUDA kernels. The first performance optimization is fusing the first Linear and GELU operations so their intermediate tensor does not make a round trip through GPU global memory.

## Phase 1 status

The initial project scope is frozen and documented:

- [Project scope](docs/project-scope.md)
- [Acceptance criteria](docs/acceptance-criteria.md)
- [Canonical benchmark workloads](config/workloads.json)

No compiler implementation exists yet. The next milestone is the CMake/MLIR
project skeleton and a `tensorforge-opt` command that can parse and print MLIR.
