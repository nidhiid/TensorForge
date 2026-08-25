# Phase 9 CUDA Runtime Lowering

Phase 9 connects compiler IR to the CUDA runtime. Before this phase,
TensorForge had CUDA code, but a compiled MLIR program could not call it.

The `--tf-lower-to-cuda-runtime` pass rewrites:

```text
tf.matmul              -> tfCudaRunMatmulHostF32
tf.linear              -> tfCudaRunLinearHostF32
tf.fused_linear_gelu   -> tfCudaRunFusedLinearGeluHostF32
```

Each call receives host addresses and the static `M`, `K`, and `N` dimensions.
The runtime wrapper creates a CUDA context, copies the tensors to the GPU,
launches the appropriate kernel, copies the result back, and reports failures
through `tfCudaCheckStatus`. Standard MLIR passes then turn `func`, buffer, and
pointer operations into LLVM dialect IR. The final IR contains ordinary
external function calls that `mlir-runner` resolves from
`libTensorForgeCudaRuntime.so`.

`tf.reshape` and `tf.transpose` lower to standard `tensor` and `linalg`
operations. A standalone `tf.gelu` is rejected because the CUDA compiler path
currently supports GELU only when the fusion pass first combines it with
Linear:

```bash
./build/bin/tensorforge-opt input.mlir \
  --tf-fuse-linear-gelu \
  --tf-lower-to-cuda-runtime
```

The lowering test checks both the runtime-call form and the final LLVM dialect
form. It runs on non-CUDA machines because it validates generated code without
launching a GPU kernel.

## Current limitation

The compiler-facing wrappers favor a simple, verifiable boundary: every
operation allocates device buffers and performs its own copies. That is correct
for static contiguous FP32 tensors, but it is not the final high-performance
design. A future buffer-planning pass should keep tensors on the GPU across
operations and reuse one context and stream.
