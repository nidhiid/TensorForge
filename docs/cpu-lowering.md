# Phase 6 CPU Lowering

Phase 6 turns TensorForge's high-level operations into standard MLIR that can
run on a CPU. Run the custom lowering pass with:

```bash
./build/bin/tensorforge-opt --tf-lower-to-linalg input.mlir
```

## Lowering rules

| TensorForge operation | Standard MLIR representation |
| --- | --- |
| `tf.matmul` | zero-filled output plus `linalg.matmul` |
| `tf.linear` | `linalg.matmul` plus a bias-broadcast `linalg.generic` |
| `tf.gelu` | elementwise `linalg.generic` containing arithmetic and `math.tanh` |
| `tf.fused_linear_gelu` | MatMul followed by one combined bias-add/GELU `linalg.generic` |
| `tf.reshape` | `tensor.reshape` with a constant target shape |
| `tf.transpose` | `linalg.transpose` with permutation `[1, 0]` |

GELU consistently uses PyTorch's tanh approximation:

```text
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

After this pass, MLIR's standard bufferization and conversion passes lower the
program through loops and memory operations to LLVM dialect IR. `mlir-runner`
then JIT-compiles that IR into native CPU instructions.

The execution test JIT-runs all six lowering rules with small, hand-checkable
constants. For example, the fused test uses an identity weight matrix and zero
bias, so the first output entering GELU is exactly `1.0`; the compiled program
must return approximately `0.841192`.

## Current boundary

This phase establishes executable CPU semantics and a correctness path. The
CPU lowering still represents MatMul and the following bias/GELU calculation
as separate standard operations. Avoiding the intermediate GPU-memory round
trip requires the later CUDA fused kernel; Phase 6 does not claim that GPU
optimization yet.
