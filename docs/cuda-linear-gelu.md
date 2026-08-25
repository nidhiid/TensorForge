# Phase 10 Fused CUDA Linear+GELU

The fused kernel computes, for every output element:

```text
output = GELU(input x weight + bias)
```

It uses the same bounds-checked 16x16 shared-memory MatMul tiles as Phase 8.
After a thread finishes its dot product, it adds the bias and applies PyTorch's
tanh GELU approximation before writing the value to global memory.

The unfused comparison launches two kernels:

```text
Linear -> write intermediate -> GELU reads intermediate -> write output
```

The fused path launches one kernel:

```text
Linear + GELU -> write output
```

Therefore fusion removes one kernel launch and avoids one write plus one read
of the `[M,N]` intermediate tensor. The benchmark reports this byte saving as a
theoretical traffic estimate. A measured DRAM-traffic claim still requires
Nsight Compute hardware counters.

## Verification and benchmark

The C++ GPU correctness test uses `M=17`, `K=19`, and `N=13`, so it also checks
partial CUDA tiles. It compares the fused kernel and separate Linear/GELU
kernels with a CPU reference.

The Python benchmark:

- loads the CUDA runtime through `ctypes`;
- uses identical FP32 inputs and weights for fused, unfused, and PyTorch eager;
- checks the fused output against PyTorch's tanh GELU;
- excludes allocation and copies from kernel timing;
- uses CUDA events, warm-up runs, repeated samples, and reports median and p95;
- evaluates the four workloads in `config/workloads.json`;
- optionally writes machine-readable JSON including GPU, CUDA, and PyTorch
  versions.

Run it after a CUDA build:

```bash
.venv/bin/python examples/benchmark_cuda.py \
  --runtime build-cuda/lib/libTensorForgeCudaRuntime.so \
  --warmup 10 --samples 20 --iterations 100 \
  --json-output artifacts/phase10/benchmark.json
```

Run the complete PyTorch-to-MLIR-to-LLVM-to-CUDA example with:

```bash
.venv/bin/python examples/run_mlp_cuda.py \
  --cuda-runtime build-cuda/lib/libTensorForgeCudaRuntime.so
```

Do not copy a speedup into a resume until the JSON report has been produced on
a named NVIDIA GPU and the output shows correctness `PASS`.
