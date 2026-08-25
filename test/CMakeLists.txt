add_test(
  NAME tensorforge-opt-parses-and-prints-mlir
  COMMAND $<TARGET_FILE:tensorforge-opt> ${CMAKE_CURRENT_SOURCE_DIR}/smoke.mlir
)

set_tests_properties(
  tensorforge-opt-parses-and-prints-mlir
  PROPERTIES PASS_REGULAR_EXPRESSION "func\\.func @identity"
)

add_test(
  NAME tensorforge-opt-rejects-invalid-mlir
  COMMAND $<TARGET_FILE:tensorforge-opt> ${CMAKE_CURRENT_SOURCE_DIR}/invalid.mlir
)

set_tests_properties(
  tensorforge-opt-rejects-invalid-mlir
  PROPERTIES WILL_FAIL TRUE
)

add_test(
  NAME tensorforge-opt-parses-tensorforge-dialect
  COMMAND $<TARGET_FILE:tensorforge-opt>
          ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/valid.mlir
)

set_tests_properties(
  tensorforge-opt-parses-tensorforge-dialect
  PROPERTIES PASS_REGULAR_EXPRESSION "tf\\.gelu"
)

add_test(
  NAME tensorforge-opt-verifies-dialect-diagnostics
  COMMAND $<TARGET_FILE:tensorforge-opt>
          --split-input-file
          --verify-diagnostics
          ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/invalid.mlir
)

add_test(
  NAME tensorforge-infers-transformer-mlp-shapes
  COMMAND $<TARGET_FILE:tensorforge-opt>
          ${CMAKE_CURRENT_SOURCE_DIR}/ShapeInference/transformer-mlp.mlir
)

set_tests_properties(
  tensorforge-infers-transformer-mlp-shapes
  PROPERTIES PASS_REGULAR_EXPRESSION "func\\.func @bert_mlp"
)

add_test(
  NAME tensorforge-canonicalizes-and-fuses-operations
  COMMAND ${CMAKE_COMMAND}
          -DTENSORFORGE_OPT=$<TARGET_FILE:tensorforge-opt>
          -DFUSION_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/Transforms/fuse-linear-gelu.mlir
          -DNO_FUSION_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/Transforms/no-fuse-multiple-users.mlir
          -DCANONICALIZE_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/Transforms/canonicalize.mlir
          -P ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/check-optimizations.cmake
)

add_test(
  NAME tensorforge-lowers-all-operations-to-cpu-dialects
  COMMAND ${CMAKE_COMMAND}
          -DTENSORFORGE_OPT=$<TARGET_FILE:tensorforge-opt>
          -DLOWERING_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/Lowering/to-linalg.mlir
          -P ${CMAKE_CURRENT_SOURCE_DIR}/Lowering/check-cpu-lowering.cmake
)

find_program(
  TENSORFORGE_MLIR_RUNNER
  NAMES mlir-runner
  HINTS "${LLVM_TOOLS_BINARY_DIR}" "${MLIR_TOOLS_DIR}"
  REQUIRED
)

add_test(
  NAME tensorforge-executes-all-operations-on-cpu
  COMMAND ${CMAKE_COMMAND}
          -DTENSORFORGE_OPT=$<TARGET_FILE:tensorforge-opt>
          -DMLIR_RUNNER=${TENSORFORGE_MLIR_RUNNER}
          -DEXECUTION_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/Execution/cpu.mlir
          -DLOWERED_FILE=${CMAKE_CURRENT_BINARY_DIR}/tensorforge-cpu-llvm.mlir
          -P ${CMAKE_CURRENT_SOURCE_DIR}/Execution/check-cpu-execution.cmake
)
