#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "TensorForge/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  tensorforge::registerTensorForgePasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  tensorforge::TensorForgeDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TensorForge MLIR optimizer driver\n", registry));
}
