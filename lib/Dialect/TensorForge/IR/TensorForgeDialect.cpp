#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
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

mlir::LogicalResult
inferMatmulReturnType(std::optional<mlir::Location> location,
                      mlir::ValueRange operands, llvm::StringRef operationName,
                      llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() < 2)
    return mlir::emitOptionalError(location, operationName,
                                   " expects two matrix operands");

  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(operands[1].getType());
  if (!lhsType || !rhsType || lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return mlir::emitOptionalError(
        location, operationName,
        " requires rank-2 tensors to infer its result type");

  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      {lhsType.getDimSize(0), rhsType.getDimSize(1)},
      lhsType.getElementType()));
  return mlir::success();
}

} // namespace

mlir::LogicalResult MatMulOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 2)
    return mlir::emitOptionalError(location,
                                   "tf.matmul expects exactly two operands");
  return inferMatmulReturnType(location, operands, "tf.matmul",
                               inferredReturnTypes);
}

mlir::LogicalResult LinearOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 3)
    return mlir::emitOptionalError(location,
                                   "tf.linear expects exactly three operands");
  return inferMatmulReturnType(location, operands, "tf.linear",
                               inferredReturnTypes);
}

mlir::LogicalResult GeluOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1)
    return mlir::emitOptionalError(location,
                                   "tf.gelu expects exactly one operand");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
  if (!inputType)
    return mlir::emitOptionalError(
        location, "tf.gelu requires a ranked tensor to infer its result type");

  inferredReturnTypes.push_back(inputType);
  return mlir::success();
}

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
