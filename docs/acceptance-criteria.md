# TensorForge v0.1 Acceptance Criteria

TensorForge v0.1 is complete only when all of the following are reproducible.

## Compilation

- A supported PyTorch MLP is captured through `torch.fx`.
- Its graph is converted into the TensorForge MLIR dialect.
- Invalid ranks, shapes, and element types produce clear diagnostics.
- Shape inference produces the expected result type for every supported op.
- The compiler visibly rewrites `linear -> gelu` to
  `fused_linear_gelu` when the linear result has no other users.
- A graph with another user of the linear result is not incorrectly fused.
- The optimized program lowers to LLVM IR containing calls into the TensorForge
  CUDA runtime.

## Correctness

- All four canonical workloads execute from Python through the compiled path.
- Fused and unfused paths are both tested.
- Results are compared against the same PyTorch model and pass a documented
  FP32 `rtol` and `atol`.
- Edge tests cover incompatible dimensions, unsupported operations, and matrix
  dimensions that are not multiples of the CUDA tile size.
- Repeated executions return stable results and do not leak device allocations.

## Performance evaluation

- PyTorch eager, TensorForge unfused, and TensorForge fused use identical
  inputs and weights.
- Each result includes warm-up iterations followed by at least 100 measured
  iterations.
- Median latency and at least one tail percentile are reported.
- GPU timing uses CUDA events and excludes compilation and one-time setup.
- Nsight Compute records DRAM read/write traffic for fused and unfused paths.
- Kernel-launch counts are reported for both paths.
- Any claimed speedup or traffic reduction is calculated from recorded results
  and names the GPU, CUDA version, PyTorch version, shapes, and baseline.

## Demonstration

A single documented command must:

1. Capture a canonical MLP.
2. Print or save input MLIR.
3. Run the optimization pipeline.
4. Show whether fusion occurred and which tile was selected.
5. Execute the CUDA result.
6. Check it against PyTorch.
7. Print benchmark measurements.

The repository must also include tests for the dialect, passes, CUDA kernels,
and the complete Python-to-GPU path.
