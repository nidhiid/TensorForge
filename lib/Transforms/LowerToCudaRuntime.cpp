#include "TensorForge/Transforms/Passes.h"

#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tensorforge {
namespace {

constexpr llvm::StringLiteral kMatmulFunction = "tfCudaRunMatmulHostF32";
constexpr llvm::StringLiteral kLinearFunction = "tfCudaRunLinearHostF32";
constexpr llvm::StringLiteral kFusedFunction =
    "tfCudaRunFusedLinearGeluHostF32";
constexpr llvm::StringLiteral kCheckFunction = "tfCudaCheckStatus";

void declareFunction(mlir::ModuleOp module, llvm::StringRef name,
                     mlir::FunctionType type) {
  if (module.lookupSymbol<mlir::func::FuncOp>(name))
    return;

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto function =
      mlir::func::FuncOp::create(builder, module.getLoc(), name, type);
  function.setPrivate();
}

void declareRuntimeFunctions(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();
  auto index = mlir::IndexType::get(context);
  auto i64 = mlir::IntegerType::get(context, 64);
  auto i32 = mlir::IntegerType::get(context, 32);

  declareFunction(module, kMatmulFunction,
                  mlir::FunctionType::get(
                      context, {index, index, index, i64, i64, i64}, {i32}));
  auto linearType = mlir::FunctionType::get(
      context, {index, index, index, index, i64, i64, i64}, {i32});
  declareFunction(module, kLinearFunction, linearType);
  declareFunction(module, kFusedFunction, linearType);
  declareFunction(module, kCheckFunction,
                  mlir::FunctionType::get(context, {i32}, {}));
}

mlir::Value tensorPointer(mlir::PatternRewriter &rewriter,
                          mlir::Location location, mlir::Value tensor,
                          bool readOnly) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(tensor.getType());
  auto bufferType =
      mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  mlir::Value buffer = mlir::bufferization::ToBufferOp::create(
      rewriter, location, bufferType, tensor, readOnly);
  return mlir::memref::ExtractAlignedPointerAsIndexOp::create(rewriter,
                                                              location, buffer);
}

mlir::Value i64Constant(mlir::PatternRewriter &rewriter,
                        mlir::Location location, int64_t value) {
  return mlir::arith::ConstantIntOp::create(rewriter, location, value, 64);
}

mlir::Value createRuntimeResult(mlir::PatternRewriter &rewriter,
                                mlir::Location location,
                                mlir::RankedTensorType resultType,
                                mlir::ValueRange inputs,
                                llvm::StringRef runtimeFunction, int64_t m,
                                int64_t k, int64_t n) {
  mlir::Value output = mlir::tensor::EmptyOp::create(
      rewriter, location, resultType.getShape(), resultType.getElementType());
  llvm::SmallVector<mlir::Value> arguments;
  arguments.reserve(inputs.size() + 4);
  for (mlir::Value input : inputs)
    arguments.push_back(tensorPointer(rewriter, location, input, true));
  arguments.push_back(tensorPointer(rewriter, location, output, false));
  arguments.push_back(i64Constant(rewriter, location, m));
  arguments.push_back(i64Constant(rewriter, location, k));
  arguments.push_back(i64Constant(rewriter, location, n));

  auto call = mlir::func::CallOp::create(rewriter, location, runtimeFunction,
                                         mlir::TypeRange{rewriter.getI32Type()},
                                         arguments);
  mlir::func::CallOp::create(rewriter, location, kCheckFunction,
                             mlir::TypeRange{}, call.getResult(0));
  return output;
}

class LowerCudaMatMulPattern : public mlir::OpRewritePattern<MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatMulOp operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto lhsType =
        mlir::cast<mlir::RankedTensorType>(operation.getLhs().getType());
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    rewriter.replaceOp(
        operation,
        createRuntimeResult(rewriter, operation.getLoc(), resultType,
                            {operation.getLhs(), operation.getRhs()},
                            kMatmulFunction, lhsType.getDimSize(0),
                            lhsType.getDimSize(1), resultType.getDimSize(1)));
    return mlir::success();
  }
};

template <typename OperationType>
class LowerCudaLinearLikePattern
    : public mlir::OpRewritePattern<OperationType> {
public:
  LowerCudaLinearLikePattern(mlir::MLIRContext *context,
                             llvm::StringRef runtimeFunction)
      : mlir::OpRewritePattern<OperationType>(context),
        runtimeFunction(runtimeFunction) {}

  mlir::LogicalResult
  matchAndRewrite(OperationType operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputType =
        mlir::cast<mlir::RankedTensorType>(operation.getInput().getType());
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    rewriter.replaceOp(
        operation,
        createRuntimeResult(
            rewriter, operation.getLoc(), resultType,
            {operation.getInput(), operation.getWeight(), operation.getBias()},
            runtimeFunction, inputType.getDimSize(0), inputType.getDimSize(1),
            resultType.getDimSize(1)));
    return mlir::success();
  }

private:
  llvm::StringRef runtimeFunction;
};

class LowerCudaReshapePattern : public mlir::OpRewritePattern<ReshapeOp> {
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

class LowerCudaTransposePattern : public mlir::OpRewritePattern<TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TransposeOp operation,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(operation.getOutput().getType());
    mlir::Value empty = mlir::tensor::EmptyOp::create(
        rewriter, operation.getLoc(), resultType.getShape(),
        resultType.getElementType());
    auto transpose = mlir::linalg::TransposeOp::create(
        rewriter, operation.getLoc(), operation.getInput(), empty,
        llvm::ArrayRef<int64_t>{1, 0});
    rewriter.replaceOp(operation, transpose->getResult(0));
    return mlir::success();
  }
};

class LowerToCudaRuntimePass
    : public mlir::PassWrapper<LowerToCudaRuntimePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToCudaRuntimePass)

  llvm::StringRef getArgument() const final {
    return "tf-lower-to-cuda-runtime";
  }

  llvm::StringRef getDescription() const final {
    return "Lower TensorForge operations to CUDA runtime calls";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    declareRuntimeFunctions(getOperation());

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerCudaMatMulPattern, LowerCudaReshapePattern,
                 LowerCudaTransposePattern>(&getContext());
    patterns.add<LowerCudaLinearLikePattern<LinearOp>>(&getContext(),
                                                       kLinearFunction);
    patterns.add<LowerCudaLinearLikePattern<FusedLinearGeluOp>>(&getContext(),
                                                                kFusedFunction);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    mlir::WalkResult unsupported = getOperation().walk([&](GeluOp operation) {
      operation.emitError(
          "standalone GELU has no CUDA lowering; run --tf-fuse-linear-gelu "
          "before --tf-lower-to-cuda-runtime");
      return mlir::WalkResult::interrupt();
    });
    if (unsupported.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerToCudaRuntimePass() {
  return std::make_unique<LowerToCudaRuntimePass>();
}

} // namespace tensorforge
