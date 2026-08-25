# Phase 8 CUDA Runtime and MatMul

Phase 8 establishes the boundary between compiled TensorForge programs and
CUDA. The shared `TensorForgeCudaRuntime` library exposes a C ABI so that later
LLVM IR can call it without depending on C++ name mangling.

## Runtime flow

A caller performs one inference operation in this order:

```text
create device context and non-blocking stream
  -> allocate device buffers
  -> queue host-to-device copies
  -> queue FP32 MatMul
  -> queue device-to-host copy
  -> synchronize the stream
  -> free buffers and destroy the context
```

The public header is `include/TensorForge/Runtime/CudaRuntime.h`. Every
operation returns a `TfCudaStatus`; `tfCudaGetLastError` provides the detailed
CUDA diagnostic stored by the context.

## MatMul kernel

`tfCudaMatmulF32` computes row-major:

```text
lhs[M,K] x rhs[K,N] -> output[M,N]
```

The initial kernel launches 16x16 thread blocks. Each block cooperatively loads
one LHS tile and one RHS tile into shared memory, synchronizes, and accumulates
the tile product. Out-of-range loads become zero and out-of-range stores are
discarded, so dimensions do not have to be divisible by 16.

The CUDA correctness test compares the GPU with a CPU reference for:

- `[2,3] x [3,4]`
- `[17,19] x [19,13]`
- `[31,7] x [7,29]`

The latter two deliberately exercise partial tiles.

## Portable builds

CUDA is optional at configuration time. With
`TENSORFORGE_ENABLE_CUDA=OFF`, CMake builds the same public API using a stub.
Device count is zero and context creation returns `TF_CUDA_ERROR_UNAVAILABLE`
with a clear diagnostic. This keeps frontend, MLIR, and runtime API tests
usable on non-NVIDIA development machines.

With `TENSORFORGE_ENABLE_CUDA=ON`, CMake enables the CUDA language, requires the
CUDA Toolkit, builds `MatMul.cu`, and links the CUDA Runtime library. See the
[official CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) and
[CMake CUDA architectures documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html).

## Later phases

Phase 9 now rewrites MLIR into calls to this runtime, and Phase 10 adds Linear,
GELU, fused Linear+GELU, CUDA-event timing, and Python evaluation. See
[CUDA lowering](cuda-lowering.md) and [fused Linear+GELU](cuda-linear-gelu.md).
The original MatMul API and test remain the small, independently testable base
of that work.

The real CUDA sources cannot be compiled or executed on Apple Silicon. The
portable stub is compiled and tested locally; the GPU correctness test must be
run on a Linux/NVIDIA CUDA machine before claiming measured CUDA correctness or
performance.
