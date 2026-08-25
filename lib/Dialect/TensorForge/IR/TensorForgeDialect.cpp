#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"

using namespace tensorforge;

#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.cpp.inc"

void TensorForgeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.cpp.inc"
      >();
}

mlir::Operation *
TensorForgeDialect::materializeConstant(mlir::OpBuilder &builder,
                                        mlir::Attribute value, mlir::Type type,
                                        mlir::Location location) {
  auto typedValue = mlir::dyn_cast<mlir::TypedAttr>(value);
  if (!typedValue || typedValue.getType() != type)
    return nullptr;

  return mlir::arith::ConstantOp::create(builder, location, type, typedValue)
      .getOperation();
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

mlir::LogicalResult verifyLinearLike(mlir::Operation *operation,
                                     mlir::RankedTensorType inputType,
                                     mlir::RankedTensorType weightType,
                                     mlir::RankedTensorType biasType,
                                     mlir::RankedTensorType outputType) {
  if (mlir::failed(verifyMatmulLike(operation, inputType, weightType,
                                    outputType, "input", "weight")))
    return mlir::failure();

  if (mlir::failed(verifyTensor(operation, biasType, "bias")))
    return mlir::failure();

  if (biasType.getRank() != 1)
    return operation->emitOpError("expects a rank-1 bias tensor, but got ")
           << biasType;

  if (biasType.getDimSize(0) != weightType.getDimSize(1))
    return operation->emitOpError(
               "bias length must equal the weight output dimension ")
           << weightType.getDimSize(1) << ", but got "
           << biasType.getDimSize(0);

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

mlir::LogicalResult
inferLinearReturnType(std::optional<mlir::Location> location,
                      mlir::ValueRange operands, llvm::StringRef operationName,
                      llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 3)
    return mlir::emitOptionalError(location, operationName,
                                   " expects exactly three operands");
  return inferMatmulReturnType(location, operands, operationName,
                               inferredReturnTypes);
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
  return inferLinearReturnType(location, operands, "tf.linear",
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

mlir::LogicalResult FusedLinearGeluOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  return inferLinearReturnType(location, operands, "tf.fused_linear_gelu",
                               inferredReturnTypes);
}

mlir::LogicalResult ReshapeOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1)
    return mlir::emitOptionalError(location,
                                   "tf.reshape expects exactly one operand");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
  auto shape = attributes.getAs<mlir::DenseI64ArrayAttr>("shape");
  if (!shape && properties)
    shape = properties.as<ReshapeOp::Properties *>()->shape;
  if (!inputType || !shape)
    return mlir::emitOptionalError(
        location, "tf.reshape requires a ranked tensor and a target shape");

  for (int64_t dimension : shape.asArrayRef()) {
    if (dimension < 0)
      return mlir::emitOptionalError(
          location, "tf.reshape target dimensions must be non-negative");
  }

  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      shape.asArrayRef(), inputType.getElementType()));
  return mlir::success();
}

mlir::LogicalResult TransposeOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::OpaqueProperties,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1)
    return mlir::emitOptionalError(location,
                                   "tf.transpose expects exactly one operand");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(operands[0].getType());
  if (!inputType || inputType.getRank() != 2)
    return mlir::emitOptionalError(
        location, "tf.transpose requires a rank-2 tensor to infer its result");

  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      {inputType.getDimSize(1), inputType.getDimSize(0)},
      inputType.getElementType()));
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

  return verifyLinearLike(*this, inputType, weightType, biasType, outputType);
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

mlir::LogicalResult FusedLinearGeluOp::verify() {
  auto inputType = mlir::cast<mlir::RankedTensorType>(getInput().getType());
  auto weightType = mlir::cast<mlir::RankedTensorType>(getWeight().getType());
  auto biasType = mlir::cast<mlir::RankedTensorType>(getBias().getType());
  auto outputType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());

  return verifyLinearLike(*this, inputType, weightType, biasType, outputType);
}

mlir::LogicalResult ReshapeOp::verify() {
  auto inputType = mlir::cast<mlir::RankedTensorType>(getInput().getType());
  auto outputType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());

  if (mlir::failed(verifyTensor(*this, inputType, "input")) ||
      mlir::failed(verifyTensor(*this, outputType, "output")))
    return mlir::failure();

  if (inputType.getNumElements() != outputType.getNumElements())
    return emitOpError("input and output must contain the same number of "
                       "elements, but got ")
           << inputType.getNumElements() << " and "
           << outputType.getNumElements();

  return mlir::success();
}

mlir::OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  if (getInput().getType() == getOutput().getType())
    return getInput();

  if (auto elements =
          mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(adaptor.getInput()))
    return elements.reshape(
        mlir::cast<mlir::RankedTensorType>(getOutput().getType()));

  return {};
}

mlir::LogicalResult TransposeOp::verify() {
  auto inputType = mlir::cast<mlir::RankedTensorType>(getInput().getType());
  auto outputType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());

  if (mlir::failed(verifyTensor(*this, inputType, "input")) ||
      mlir::failed(verifyTensor(*this, outputType, "output")))
    return mlir::failure();

  if (inputType.getRank() != 2 || outputType.getRank() != 2)
    return emitOpError("expects rank-2 input and output tensors");

  if (outputType.getDimSize(0) != inputType.getDimSize(1) ||
      outputType.getDimSize(1) != inputType.getDimSize(0))
    return emitOpError("output must transpose the input shape, but got ")
           << inputType << " and " << outputType;

  return mlir::success();
}

namespace {

class CollapseReshapePattern : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReshapeOp reshape,
                  mlir::PatternRewriter &rewriter) const override {
    auto inner = reshape.getInput().getDefiningOp<ReshapeOp>();
    if (!inner)
      return mlir::failure();

    rewriter.modifyOpInPlace(reshape,
                             [&] { reshape->setOperand(0, inner.getInput()); });
    return mlir::success();
  }
};

class CancelDoubleTransposePattern
    : public mlir::OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TransposeOp transpose,
                  mlir::PatternRewriter &rewriter) const override {
    auto inner = transpose.getInput().getDefiningOp<TransposeOp>();
    if (!inner)
      return mlir::failure();

    rewriter.replaceOp(transpose, inner.getInput());
    return mlir::success();
  }
};

} // namespace

void ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::MLIRContext *context) {
  patterns.add<CollapseReshapePattern>(context);
}

void TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::MLIRContext *context) {
  patterns.add<CancelDoubleTransposePattern>(context);
}

#define GET_OP_CLASSES
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.cpp.inc"
