include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TensorForge_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorForge_Dialect, mnemonic, traits>;

def TensorForge_MatMulOp : TensorForge_Op<"matmul", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Multiply two rank-2 tensors";
  let description = [{
    Multiplies a static FP32 tensor of shape `[M, K]` by one of shape `[K, N]`
    and produces a tensor of shape `[M, N]`.
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def TensorForge_LinearOp : TensorForge_Op<"linear", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
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
    type($weight) `,` type($bias)
  }];
}

def TensorForge_GeluOp : TensorForge_Op<"gelu", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Apply GELU elementwise";
  let description = [{
    Applies the GELU activation to every element of a static FP32 tensor. The
    output has exactly the same type and shape as the input.
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def TensorForge_FusedLinearGeluOp : TensorForge_Op<"fused_linear_gelu", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Apply a linear layer and GELU as one fused operation";
  let description = [{
    Represents `gelu(input * weight + bias)` as one compiler operation. Its
    shape and validation rules are the same as `tf.linear`; later lowering can
    map it to a fused CUDA kernel that avoids an intermediate global-memory
    round trip.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$weight,
                       AnyRankedTensor:$bias);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict `:` type($input) `,`
    type($weight) `,` type($bias)
  }];
}

def TensorForge_ReshapeOp : TensorForge_Op<"reshape", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Change a tensor's static shape without changing its elements";
  let description = [{
    Reinterprets a static FP32 tensor with a new static shape. The input and
    output must contain the same number of elements.
  }];

  let arguments = (ins AnyRankedTensor:$input, DenseI64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $input `to` $shape attr-dict `:` type($input)
  }];
}

def TensorForge_TransposeOp : TensorForge_Op<"transpose", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Transpose a rank-2 tensor";
  let description = [{
    Swaps the two dimensions of a static rank-2 FP32 tensor. Applying this
    operation twice returns the original value and is canonicalized away.
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}
