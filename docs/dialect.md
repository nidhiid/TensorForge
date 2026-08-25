# TensorForge Dialect

The `tf` dialect is TensorForge's high-level MLIR vocabulary. Phase 3 defines
three pure tensor operations. They describe computation but do not execute it
or lower it to loops, LLVM IR, or CUDA yet.

## `tf.matmul`

Multiplies `[M, K]` by `[K, N]` and returns `[M, N]`:

```mlir
%result = tf.matmul %lhs, %rhs
    : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
```

## `tf.linear`

Computes `input * weight + bias`. The bias is broadcast over the first output
dimension:

```mlir
%result = tf.linear %input, %weight, %bias
    : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
      -> tensor<2x4xf32>
```

## `tf.gelu`

Describes elementwise GELU and preserves the input type:

```mlir
%result = tf.gelu %input : tensor<2x4xf32> -> tensor<2x4xf32>
```

## Version 0.1 verification rules

- Every operand and result must be a ranked tensor.
- Every shape must be fully static.
- Every element type must be `f32`.
- MatMul contracting dimensions must match.
- Linear input, weight, bias, and result dimensions must agree.
- GELU input and result types must be identical.

Phase 3 requires result types in the input MLIR and checks that they are right.
Phase 4 will infer result shapes automatically.
