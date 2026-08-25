include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TensorForge_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorForge_Dialect, mnemonic, traits>;

def TensorForge_MatMulOp : TensorForge_Op<"matmul", [Pure]> {
  let summary = "Multiply two rank-2 tensors";
  let description = [{
    Multiplies a static FP32 tensor of shape `[M, K]` by one of shape `[K, N]`
    and produces a tensor of shape `[M, N]`.
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
}

def TensorForge_LinearOp : TensorForge_Op<"linear", [Pure]> {
  let summary = "Apply a rank-2 linear layer with a rank-1 bias";
  let description = [{
    Multiplies an input of shape `[M, K]` by a weight of shape `[K, N]`, then
    broadcasts and adds a bias of shape `[N]`. The output shape is `[M, N]`.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$weight,
                       AnyRankedTensor:$bias);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict `:` type($input) `,`
    type($weight) `,` type($bias) `->` type($output)
  }];
}

def TensorForge_GeluOp : TensorForge_Op<"gelu", [Pure]> {
  let summary = "Apply GELU elementwise";
  let description = [{
    Applies the GELU activation to every element of a static FP32 tensor. The
    output has exactly the same type and shape as the input.
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}
