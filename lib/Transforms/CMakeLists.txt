add_mlir_library(TensorForgeTransforms
  FuseLinearGelu.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  TensorForgeDialect
)
