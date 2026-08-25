add_mlir_library(TensorForgeTransforms
  FuseLinearGelu.cpp
  LowerToLinalg.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  TensorForgeDialect
)
