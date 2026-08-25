execute_process(
  COMMAND "${TENSORFORGE_OPT}" "${LOWERING_INPUT}"
          --tf-fuse-linear-gelu --tf-lower-to-cuda-runtime
  RESULT_VARIABLE lowering_result
  OUTPUT_VARIABLE lowered_ir
  ERROR_VARIABLE lowering_error
)

if(NOT lowering_result EQUAL 0)
  message(FATAL_ERROR "CUDA runtime lowering failed:\n${lowering_error}")
endif()

foreach(expected_symbol
    "tfCudaRunMatmulHostF32"
    "tfCudaRunFusedLinearGeluHostF32"
    "tfCudaRunLinearHostF32"
    "tfCudaCheckStatus"
    "tensor.reshape"
    "linalg.transpose")
  string(FIND "${lowered_ir}" "${expected_symbol}" symbol_position)
  if(symbol_position EQUAL -1)
    message(FATAL_ERROR
            "Expected ${expected_symbol} in CUDA-lowered IR:\n${lowered_ir}")
  endif()
endforeach()

string(FIND "${lowered_ir}" "tf." tensorforge_operation_position)
if(NOT tensorforge_operation_position EQUAL -1)
  message(FATAL_ERROR
          "TensorForge operations remained after CUDA lowering:\n${lowered_ir}")
endif()

execute_process(
  COMMAND "${TENSORFORGE_OPT}" "${LOWERING_INPUT}"
          --tf-fuse-linear-gelu --tf-lower-to-cuda-runtime
          --one-shot-bufferize=bufferize-function-boundaries
          --convert-bufferization-to-memref --convert-linalg-to-loops
          --convert-scf-to-cf --expand-strided-metadata
          --convert-arith-to-llvm --convert-cf-to-llvm
          --convert-index-to-llvm --finalize-memref-to-llvm
          --convert-func-to-llvm --reconcile-unrealized-casts
  RESULT_VARIABLE llvm_result
  OUTPUT_VARIABLE llvm_ir
  ERROR_VARIABLE llvm_error
)

if(NOT llvm_result EQUAL 0)
  message(FATAL_ERROR "CUDA-to-LLVM lowering failed:\n${llvm_error}")
endif()

foreach(expected_symbol
    "llvm.call @tfCudaRunMatmulHostF32"
    "llvm.call @tfCudaRunFusedLinearGeluHostF32"
    "llvm.call @tfCudaRunLinearHostF32")
  string(FIND "${llvm_ir}" "${expected_symbol}" symbol_position)
  if(symbol_position EQUAL -1)
    message(FATAL_ERROR "Expected ${expected_symbol} in LLVM IR:\n${llvm_ir}")
  endif()
endforeach()

execute_process(
  COMMAND "${TENSORFORGE_OPT}" "${STANDALONE_GELU_INPUT}"
          --tf-lower-to-cuda-runtime
  RESULT_VARIABLE gelu_result
  ERROR_VARIABLE gelu_error
)

if(gelu_result EQUAL 0)
  message(FATAL_ERROR "Standalone GELU unexpectedly received a CUDA lowering")
endif()

string(FIND "${gelu_error}" "run --tf-fuse-linear-gelu" diagnostic_position)
if(diagnostic_position EQUAL -1)
  message(FATAL_ERROR "Missing standalone GELU diagnostic:\n${gelu_error}")
endif()
