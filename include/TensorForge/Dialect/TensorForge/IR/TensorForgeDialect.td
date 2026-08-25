include "mlir/IR/OpBase.td"

def TensorForge_Dialect : Dialect {
  let name = "tf";
  let cppNamespace = "::tensorforge";
  let summary = "Tensor operations used by the TensorForge compiler";
  let description = [{
    The TensorForge dialect represents the small set of high-level tensor
    operations compiled by TensorForge. Version 0.1 starts with matrix
    multiplication, linear layers, and GELU activation for static FP32
    transformer MLP workloads.
  }];
}
