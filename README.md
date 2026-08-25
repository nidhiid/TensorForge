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

Phase 2 adds the first buildable compiler component: a C++17/CMake project and
`tensorforge-opt`, a minimal MLIR command-line tool that parses and prints the
standard `func` dialect. It does not contain TensorForge operations or
optimization passes yet.

See [Building TensorForge](docs/building.md) for dependency, build, test, and
execution instructions. The next milestone is the custom TensorForge MLIR
dialect.
