# Phase 5 Optimizations

Phase 5 transforms valid TensorForge IR into simpler or more hardware-friendly
IR. Phase 6 can execute the result on a CPU; CUDA kernels remain a later phase.

## Linear-GELU fusion

Run the fusion pass with:

```bash
./build/bin/tensorforge-opt --tf-fuse-linear-gelu input.mlir
```

It rewrites:

```text
tf.linear -> tf.gelu
```

to:

```text
tf.fused_linear_gelu
```

Fusion occurs only when GELU is the Linear result's only user. If another
operation needs the intermediate Linear value, the pair remains unfused. This
prevents duplicated MatMul work and preserves the original program behavior.

## Canonicalization

Run MLIR's canonicalizer with:

```bash
./build/bin/tensorforge-opt --canonicalize input.mlir
```

TensorForge contributes patterns and fold hooks that:

- Replace an identity reshape with its input.
- Collapse consecutive reshapes.
- Remove two consecutive rank-2 transposes.
- Fold a reshape of a constant tensor into a new constant with the target
  shape.

The constant fold uses the TensorForge dialect's constant materializer to emit
an `arith.constant` operation at compile time.
