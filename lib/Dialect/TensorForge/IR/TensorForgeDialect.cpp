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

mlir::LogicalResult verifyMatmulLike(mlir::Operation *operation,
                                     mlir::RankedTensorType lhsType,
                                     mlir::RankedTensorType rhsType,
                                     mlir::RankedTensorType outputType,
                                     llvm::StringRef lhsName,
                                     llvm::StringRef rhsName) {
  if (mlir::failed(verifyTensor(operation, lhsType, lhsName)) ||
      mlir::failed(verifyTensor(operation, rhsType, rhsName)) ||
      mlir::failed(verifyTensor(operation, outputType, "output")))
    return mlir::failure();

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      outputType.getRank() != 2)
    return operation->emitOpError() << "expects rank-2 " << lhsName << ", "
                                    << rhsName << ", and output tensors";

  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return operation->emitOpError("has incompatible contracting dimensions: ")
           << lhsType.getDimSize(1) << " and " << rhsType.getDimSize(0);

  if (outputType.getDimSize(0) != lhsType.getDimSize(0) ||
      outputType.getDimSize(1) != rhsType.getDimSize(1))
    return operation->emitOpError("output must have shape [")
           << lhsType.getDimSize(0) << ", " << rhsType.getDimSize(1)
           << "], but got " << outputType;

  return mlir::success();
}

} // namespace

mlir::LogicalResult MatMulOp::verify() {
  auto lhsType = mlir::cast<mlir::RankedTensorType>(getLhs().getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(getRhs().getType());
  auto outputType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());

  return verifyMatmulLike(*this, lhsType, rhsType, outputType, "lhs", "rhs");
}

mlir::LogicalResult LinearOp::verify() {
  auto inputType = mlir::cast<mlir::RankedTensorType>(getInput().getType());
  auto weightType = mlir::cast<mlir::RankedTensorType>(getWeight().getType());
  auto biasType = mlir::cast<mlir::RankedTensorType>(getBias().getType());
  auto outputType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());

  if (mlir::failed(verifyMatmulLike(*this, inputType, weightType, outputType,
                                    "input", "weight")))
    return mlir::failure();

  if (mlir::failed(verifyTensor(*this, biasType, "bias")))
    return mlir::failure();

  if (biasType.getRank() != 1)
    return emitOpError("expects a rank-1 bias tensor, but got ") << biasType;

  if (biasType.getDimSize(0) != weightType.getDimSize(1))
    return emitOpError("bias length must equal the weight output dimension ")
           << weightType.getDimSize(1) << ", but got "
           << biasType.getDimSize(0);

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
