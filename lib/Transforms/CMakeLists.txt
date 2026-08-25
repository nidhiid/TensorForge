add_mlir_library(TensorForgeTransforms
  FuseLinearGelu.cpp
  LowerToCudaRuntime.cpp
  LowerToLinalg.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  TensorForgeDialect
)
