#include "TensorForge/Transforms/Passes.h"

#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tensorforge {
namespace {

class FuseLinearGeluPattern : public mlir::OpRewritePattern<GeluOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(GeluOp gelu, mlir::PatternRewriter &rewriter) const override {
    auto linear = gelu.getInput().getDefiningOp<LinearOp>();
    if (!linear || !linear.getOutput().hasOneUse())
      return mlir::failure();

    auto fused =
        FusedLinearGeluOp::create(rewriter, gelu.getLoc(), linear.getInput(),
                                  linear.getWeight(), linear.getBias());
    rewriter.replaceOp(gelu, fused.getOutput());
    rewriter.eraseOp(linear);
    return mlir::success();
  }
};

class FuseLinearGeluPass
    : public mlir::PassWrapper<FuseLinearGeluPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseLinearGeluPass)

  llvm::StringRef getArgument() const final { return "tf-fuse-linear-gelu"; }

  llvm::StringRef getDescription() const final {
    return "Fuse a single-use tf.linear followed by tf.gelu";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TensorForgeDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuseLinearGeluPattern>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFuseLinearGeluPass() {
  return std::make_unique<FuseLinearGeluPass>();
}

void registerTensorForgePasses() {
  mlir::registerPass([] { return createFuseLinearGeluPass(); });
  mlir::registerPass([] { return createLowerToLinalgPass(); });
}

} // namespace tensorforge
