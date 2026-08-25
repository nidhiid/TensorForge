#include "TensorForge/Transforms/Passes.h"

#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"

namespace tensorforge {
namespace {

mlir::Value createEmptyTensor(mlir::PatternRewriter &rewriter,
                              mlir::Location location,
                              mlir::RankedTensorType type) {
  return mlir::tensor::EmptyOp::create(rewriter, location, type.getShape(),
                                       type.getElementType());
}

mlir::Value createF32Constant(mlir::OpBuilder &builder, mlir::Location location,
                              float value) {
  return mlir::arith::ConstantFloatOp::create(
      builder, location, builder.getF32Type(), llvm::APFloat(value));
}

mlir::Value createMatmul(mlir::PatternRewriter &rewriter,
                         mlir::Location location, mlir::Value lhs,
                         mlir::Value rhs, mlir::RankedTensorType resultType) {
  mlir::Value empty = createEmptyTensor(rewriter, location, resultType);
  mlir::Value zero = createF32Constant(rewriter, location, 0.0f);
  auto initialized = mlir::linalg::FillOp::create(
      rewriter, location, mlir::ValueRange{zero}, mlir::ValueRange{empty});
  auto matmul = mlir::linalg::MatmulOp::create(
      rewriter, location, mlir::ValueRange{lhs, rhs},
      mlir::ValueRange{initialized.getResult(0)});
  return matmul.getResult(0);
}

mlir::Value createGeluScalar(mlir::OpBuilder &builder, mlir::Location location,
                             mlir::Value input) {
  // tanh approximation used by PyTorch's GELU(approximate="tanh"):
  // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
  mlir::Value half = createF32Constant(builder, location, 0.5f);
  mlir::Value one = createF32Constant(builder, location, 1.0f);
  mlir::Value coefficient =
      createF32Constant(builder, location, 0.7978845608028654f);
  mlir::Value cubicCoefficient =
      createF32Constant(builder, location, 0.044715f);

  mlir::Value squared =
      mlir::arith::MulFOp::create(builder, location, input, input);
  mlir::Value cubed =
      mlir::arith::MulFOp::create(builder, location, squared, input);
  mlir::Value scaledCube =
      mlir::arith::MulFOp::create(builder, location, cubicCoefficient, cubed);
  mlir::Value inner =
      mlir::arith::AddFOp::create(builder, location, input, scaledCube);
  mlir::Value scaled =
      mlir::arith::MulFOp::create(builder, location, coefficient, inner);
  mlir::Value tanh = mlir::math::TanhOp::create(builder, location, scaled);
  mlir::Value shifted =
      mlir::arith::AddFOp::create(builder, location, one, tanh);
  mlir::Value halfInput =
      mlir::arith::MulFOp::create(builder, location, half, input);
  return mlir::arith::MulFOp::create(builder, location, halfInput, shifted);
}

mlir::Value createLinearResult(mlir::PatternRewriter &rewriter,
                               mlir::Location location, mlir::Value input,
                               mlir::Value weight, mlir::Value bias,
                               mlir::RankedTensorType resultType,
                               bool applyGelu) {
  mlir::Value product =
      createMatmul(rewriter, location, input, weight, resultType);
  mlir::Value empty = createEmptyTensor(rewriter, location, resultType);

  mlir::AffineMap matrixMap =
      mlir::AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
  mlir::AffineExpr row;
  mlir::AffineExpr column;
  bindDims(rewriter.getContext(), row, column);
  mlir::AffineMap biasMap =
      mlir::AffineMap::get(2, 0, {column}, rewriter.getContext());

  auto generic = mlir::linalg::GenericOp::create(
      rewriter, location, mlir::TypeRange{resultType},
      mlir::ValueRange{product, bias}, mlir::ValueRange{empty},
      llvm::ArrayRef<mlir::AffineMap>{matrixMap, biasMap, matrixMap},
      llvm::ArrayRef<mlir::utils::IteratorType>{
          mlir::utils::IteratorType::parallel,
          mlir::utils::IteratorType::parallel},
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
          mlir::ValueRange arguments) {
        mlir::Value biased = mlir::arith::AddFOp::create(
            nestedBuilder, nestedLocation, arguments[0], arguments[1]);
        mlir::Value result =
            applyGelu ? createGeluScalar(nestedBuilder, nestedLocation, biased)
                      : biased;
        mlir::linalg::YieldOp::create(nestedBuilder, nestedLocation, result);
      });
  return generic.getResult(0);
}

class LowerMatMulPattern : public mlir::OpRewritePattern<MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatMulOp operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    rewriter.replaceOp(operation, createMatmul(rewriter, operation.getLoc(),
                                               operation.getLhs(),
                                               operation.getRhs(), resultType));
    return mlir::success();
  }
};

class LowerLinearPattern : public mlir::OpRewritePattern<LinearOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LinearOp operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    rewriter.replaceOp(
        operation,
        createLinearResult(rewriter, operation.getLoc(), operation.getInput(),
                           operation.getWeight(), operation.getBias(),
                           resultType, false));
    return mlir::success();
  }
};

class LowerGeluPattern : public mlir::OpRewritePattern<GeluOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(GeluOp operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    mlir::Value empty =
        createEmptyTensor(rewriter, operation.getLoc(), resultType);
    mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        resultType.getRank(), rewriter.getContext());
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        resultType.getRank(), mlir::utils::IteratorType::parallel);

    auto generic = mlir::linalg::GenericOp::create(
        rewriter, operation.getLoc(), mlir::TypeRange{resultType},
        mlir::ValueRange{operation.getInput()}, mlir::ValueRange{empty},
        llvm::ArrayRef<mlir::AffineMap>{identityMap, identityMap},
        iteratorTypes,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLocation,
            mlir::ValueRange arguments) {
          mlir::Value result =
              createGeluScalar(nestedBuilder, nestedLocation, arguments[0]);
          mlir::linalg::YieldOp::create(nestedBuilder, nestedLocation, result);
        });
    rewriter.replaceOp(operation, generic.getResult(0));
    return mlir::success();
  }
};

class LowerFusedLinearGeluPattern
    : public mlir::OpRewritePattern<FusedLinearGeluOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(FusedLinearGeluOp operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    rewriter.replaceOp(
        operation,
        createLinearResult(rewriter, operation.getLoc(), operation.getInput(),
                           operation.getWeight(), operation.getBias(),
                           resultType, true));
    return mlir::success();
  }
};

class LowerReshapePattern : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReshapeOp operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    auto shapeType = mlir::RankedTensorType::get({resultType.getRank()},
                                                 rewriter.getI64Type());
    auto shapeAttribute =
        mlir::DenseIntElementsAttr::get(shapeType, resultType.getShape());
    mlir::Value shape = mlir::arith::ConstantOp::create(
        rewriter, operation.getLoc(), shapeAttribute);
    rewriter.replaceOp(operation, mlir::tensor::ReshapeOp::create(
                                      rewriter, operation.getLoc(), resultType,
                                      operation.getInput(), shape));
    return mlir::success();
  }
};

class LowerTransposePattern : public mlir::OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TransposeOp operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    mlir::Value empty =
        createEmptyTensor(rewriter, operation.getLoc(), resultType);
    auto transpose = mlir::linalg::TransposeOp::create(
        rewriter, operation.getLoc(), operation.getInput(), empty,
        llvm::ArrayRef<int64_t>{1, 0});
    rewriter.replaceOp(operation, transpose->getResult(0));
    return mlir::success();
  }
};

class LowerToLinalgPass
    : public mlir::PassWrapper<LowerToLinalgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLinalgPass)

  llvm::StringRef getArgument() const final { return "tf-lower-to-linalg"; }

  llvm::StringRef getDescription() const final {
    return "Lower TensorForge operations to standard CPU-compatible MLIR";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::linalg::LinalgDialect,
                    mlir::math::MathDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerMatMulPattern, LowerLinearPattern, LowerGeluPattern,
                 LowerFusedLinearGeluPattern, LowerReshapePattern,
                 LowerTransposePattern>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerToLinalgPass() {
  return std::make_unique<LowerToLinalgPass>();
}

} // namespace tensorforge
