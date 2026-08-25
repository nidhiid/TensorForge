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
