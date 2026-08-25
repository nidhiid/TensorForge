execute_process(
  COMMAND "${TENSORFORGE_OPT}" "${EXECUTION_INPUT}"
          --tf-fuse-linear-gelu
          --tf-lower-to-linalg
          --one-shot-bufferize=bufferize-function-boundaries
          --convert-linalg-to-loops
          --convert-scf-to-cf
          --convert-math-to-llvm
          --expand-strided-metadata
          --convert-arith-to-llvm
          --convert-cf-to-llvm
          --convert-index-to-llvm
          --finalize-memref-to-llvm
          --convert-func-to-llvm
          --reconcile-unrealized-casts
          -o "${LOWERED_FILE}"
  RESULT_VARIABLE compile_result
  ERROR_VARIABLE compile_error
)

if(NOT compile_result EQUAL 0)
  message(FATAL_ERROR "CPU compilation failed:\n${compile_error}")
endif()

function(check_result entry_point expected_pattern description)
  execute_process(
    COMMAND "${MLIR_RUNNER}" "${LOWERED_FILE}"
            -e "${entry_point}" --entry-point-result=f32
    RESULT_VARIABLE execution_result
    OUTPUT_VARIABLE execution_output
    ERROR_VARIABLE execution_error
  )

  if(NOT execution_result EQUAL 0)
    message(FATAL_ERROR
            "CPU execution of ${entry_point} failed:\n${execution_error}")
  endif()

  if(NOT execution_output MATCHES "${expected_pattern}")
    message(FATAL_ERROR
            "Expected ${description}, got: ${execution_output}")
  endif()
endfunction()

check_result(run_matmul "5\\.000000e\\+01" "MatMul result 50.0")
check_result(run_linear "1\\.500000e\\+00" "Linear result 1.5")
check_result(run_gelu "8\\.4119[0-9]*e-01" "GELU(1.0) near 0.84119")
check_result(run_reshape "5\\.000000e\\+00" "reshaped value 5.0")
check_result(run_transpose "6\\.000000e\\+00" "transposed value 6.0")
check_result(main "8\\.4119[0-9]*e-01"
             "fused Linear+GELU result near 0.84119")
