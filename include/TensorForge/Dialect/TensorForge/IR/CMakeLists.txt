set(LLVM_TARGET_DEFINITIONS TensorForgeDialect.td)
mlir_tablegen(TensorForgeDialect.h.inc -gen-dialect-decls -dialect=tf)
mlir_tablegen(TensorForgeDialect.cpp.inc -gen-dialect-defs -dialect=tf)
add_public_tablegen_target(TensorForgeDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TensorForgeOps.td)
mlir_tablegen(TensorForgeOps.h.inc -gen-op-decls)
mlir_tablegen(TensorForgeOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TensorForgeOpsIncGen)
