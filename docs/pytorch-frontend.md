# Phase 7 PyTorch Frontend

Phase 7 accepts a small PyTorch transformer MLP instead of requiring the user
to write MLIR manually:

```text
PyTorch model
  -> torch.fx graph
  -> TensorForge MLIR
  -> Linear+GELU fusion
  -> CPU/LLVM lowering
  -> native execution
  -> comparison with PyTorch
```

Run the complete example with:

```bash
.venv/bin/python examples/run_mlp.py
```

The command saves three useful files in `artifacts/phase7/`:

- `input.mlir`: MLIR generated from the captured PyTorch model.
- `optimized.mlir`: IR after canonicalization and Linear+GELU fusion.
- `lowered-llvm.mlir`: CPU-ready LLVM dialect IR executed by `mlir-runner`.

## Supported model

The frontend intentionally accepts exactly:

```text
nn.Linear -> nn.GELU(approximate="tanh") -> nn.Linear
```

The input must be a concrete, static FP32 tensor with shape `[M,H]` or
`[B,S,H]`. A rank-3 input is flattened to `[B*S,H]` for TensorForge's rank-2
Linear operation, then reshaped back to `[B,S,H]` before returning the result.

PyTorch stores a Linear weight as `[output,input]`, while TensorForge computes
`[M,input] x [input,output]`. The frontend transposes every weight once while
exporting it. Bias-free PyTorch Linear modules receive an equivalent zero bias.

The second Linear must restore the original hidden size. Unsupported
activations, extra graph operations, non-FP32 inputs, and GELU's exact mode
produce frontend errors instead of silently compiling a different program.

## Correctness

The generated module prints its complete output tensor through MLIR's runner
utilities. Python reads those values and checks each one against the output of
the captured PyTorch graph using `rtol=1e-5` and `atol=1e-6`.

This is a concrete-input correctness frontend: inputs and parameters are
embedded as MLIR constants. Runtime tensor arguments, arbitrary PyTorch models,
the four large benchmark workloads, and CUDA execution remain later work.
