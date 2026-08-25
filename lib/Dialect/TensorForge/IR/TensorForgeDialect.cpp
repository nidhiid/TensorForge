#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringRef.h"

using namespace tensorforge;

#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.cpp.inc"

void TensorForgeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.cpp.inc"
      >();
}

namespace {

mlir::LogicalResult verifyTensor(mlir::Operation *operation,
                                 mlir::RankedTensorType type,
                                 llvm::StringRef valueName) {
  if (!type.hasStaticShape())
    return operation->emitOpError()
           << valueName << " must have a static shape, but got " << type;

  if (!type.getElementType().isF32())
    return operation->emitOpError()
           << valueName << " must contain f32 elements, but got " << type;

  return mlir::success();
}

} // namespace

mlir::LogicalResult MatMulOp::verify() {
  auto lhsType = mlir::cast<mlir::RankedTensorType>(getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(getRhs().getType());
  auto outputType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());

  if (mlir::failed(verifyTensor(*this, lhsType, "lhs")) ||
      mlir::failed(verifyTensor(*this, rhsType, "rhs")) ||
      mlir::failed(verifyTensor(*this, outputType, "output")))
    return mlir::failure();

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      outputType.getRank() != 2)
    return emitOpError("expects rank-2 lhs, rhs, and output tensors");

  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return emitOpError("has incompatible contracting dimensions: ")
           << lhsType.getDimSize(1) << " and " << rhsType.getDimSize(0);

  if (outputType.getDimSize(0) != lhsType.getDimSize(0) ||
      outputType.getDimSize(1) != rhsType.getDimSize(1))
    return emitOpError("output must have shape [")
           << lhsType.getDimSize(0) << ", " << rhsType.getDimSize(1)
           << "], but got " << outputType;

  return mlir::success();
}

mlir::LogicalResult LinearOp::verify() {
  auto inputType = mlir::cast<mlir::RankedTensorType>(getInput().getType());
  auto weightType = mlir::cast<mlir::RankedTensorType>(getWeight().getType());
  auto biasType = mlir::cast<mlir::RankedTensorType>(getBias().getType());
  auto outputType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());

  if (mlir::failed(verifyTensor(*this, inputType, "input")) ||
      mlir::failed(verifyTensor(*this, weightType, "weight")) ||
      mlir::failed(verifyTensor(*this, biasType, "bias")) ||
      mlir::failed(verifyTensor(*this, outputType, "output")))
    return mlir::failure();

  if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
      biasType.getRank() != 1 || outputType.getRank() != 2)
    return emitOpError(
        "expects rank-2 input, rank-2 weight, rank-1 bias, and rank-2 output");

  if (inputType.getDimSize(1) != weightType.getDimSize(0))
    return emitOpError("has incompatible input and weight dimensions: ")
           << inputType.getDimSize(1) << " and " << weightType.getDimSize(0);

  if (biasType.getDimSize(0) != weightType.getDimSize(1))
    return emitOpError("bias length must equal the weight output dimension ")
           << weightType.getDimSize(1) << ", but got "
           << biasType.getDimSize(0);

  if (outputType.getDimSize(0) != inputType.getDimSize(0) ||
      outputType.getDimSize(1) != weightType.getDimSize(1))
    return emitOpError("output must have shape [")
           << inputType.getDimSize(0) << ", " << weightType.getDimSize(1)
           << "], but got " << outputType;

  return mlir::success();
}

mlir::LogicalResult GeluOp::verify() {
  auto inputType = mlir::cast<mlir::RankedTensorType>(getInput().getType());
  auto outputType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());

  if (mlir::failed(verifyTensor(*this, inputType, "input")) ||
      mlir::failed(verifyTensor(*this, outputType, "output")))
    return mlir::failure();

  if (inputType != outputType)
    return emitOpError("input and output types must match, but got ")
           << inputType << " and " << outputType;

  return mlir::success();
}

#define GET_OP_CLASSES
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.cpp.inc"
