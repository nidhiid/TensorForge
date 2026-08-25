#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "TensorForge/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  tensorforge::registerTensorForgePasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<tensorforge::TensorForgeDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TensorForge MLIR optimizer driver\n", registry));
}
