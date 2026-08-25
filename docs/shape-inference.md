# Shape Inference

TensorForge operations implement MLIR's `InferTypeOpInterface`. The parser and
C++ builders use this interface to create result types from operand types, so a
result type does not need to be repeated in the custom operation syntax.

## Rules

- `tf.matmul`: `[M, K] x [K, N]` produces `[M, N]`.
- `tf.linear`: input `[M, K]`, weight `[K, N]`, and bias `[N]` produce `[M, N]`.
- `tf.gelu`: the result has exactly the input type.

For example:

```mlir
%result = tf.matmul %lhs, %rhs : tensor<2x3xf32>, tensor<3x4xf32>
```

The operation is created with result type `tensor<2x4xf32>`. Phase 3's
verifiers still run afterward, so incompatible contracting dimensions,
incorrect bias sizes, dynamic shapes, and non-FP32 tensors remain errors.
