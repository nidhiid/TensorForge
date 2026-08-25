#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, tensorforge::TensorForgeDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "TensorForge MLIR optimizer driver\n", registry));
}
