add_mlir_dialect_library(TensorForgeDialect
  TensorForgeDialect.cpp

  DEPENDS
  TensorForgeDialectIncGen
  TensorForgeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
)
