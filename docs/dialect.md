# TensorForge Dialect

The `tf` dialect is TensorForge's high-level MLIR vocabulary. Its operations
describe computation without committing to a specific machine.
The Phase 6 CPU path lowers these operations through standard MLIR dialects to LLVM IR;
CUDA lowering remains a later milestone.

## `tf.matmul`

Multiplies `[M, K]` by `[K, N]` and returns `[M, N]`:

```mlir
%result = tf.matmul %lhs, %rhs
    : tensor<2x3xf32>, tensor<3x4xf32>
```

## `tf.linear`

Computes `input * weight + bias`. The bias is broadcast over the first output dimension:

```mlir
%result = tf.linear %input, %weight, %bias
    : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
```

## `tf.gelu`

Describes elementwise GELU and preserves the input type:

```mlir
%result = tf.gelu %input : tensor<2x4xf32>
```

## `tf.fused_linear_gelu`

Represents Linear and GELU as one operation after the Phase 5 fusion pass:

```mlir
%result = tf.fused_linear_gelu %input, %weight, %bias
    : tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>
```

## `tf.reshape`

Changes a tensor's static shape while preserving its element count:

```mlir
%result = tf.reshape %input to [2, 3] : tensor<6xf32>
```

## `tf.transpose`

Swaps the two dimensions of a rank-2 tensor:

```mlir
%result = tf.transpose %input : tensor<3x4xf32>
```

## Version 0.1 verification rules

- Every operand and result must be a ranked tensor.
- Every shape must be fully static.
- Every element type must be `f32`.
- MatMul contracting dimensions must match.
- Linear input, weight, bias, and result dimensions must agree.
- GELU input and result types must be identical.
- Reshape input and output element counts must match.
- Transpose is limited to rank-2 tensors.

The result types are omitted from this syntax because TensorForge infers them
from the operands. In the examples above, MatMul and Linear infer
`tensor<2x4xf32>`, while GELU preserves its input type.
