execute_process(
  COMMAND "${TENSORFORGE_OPT}" "${LOWERING_INPUT}" --tf-lower-to-linalg
  RESULT_VARIABLE lowering_result
  OUTPUT_VARIABLE lowered_ir
  ERROR_VARIABLE lowering_error
)

if(NOT lowering_result EQUAL 0)
  message(FATAL_ERROR "CPU lowering failed:\n${lowering_error}")
endif()

foreach(expected_operation
    "linalg.matmul"
    "linalg.generic"
    "math.tanh"
    "tensor.reshape"
    "linalg.transpose")
  string(FIND "${lowered_ir}" "${expected_operation}" operation_position)
  if(operation_position EQUAL -1)
    message(FATAL_ERROR
            "Expected ${expected_operation} in lowered IR:\n${lowered_ir}")
  endif()
endforeach()

string(FIND "${lowered_ir}" "tf." tensorforge_operation_position)
if(NOT tensorforge_operation_position EQUAL -1)
  message(FATAL_ERROR
          "TensorForge operations remained after CPU lowering:\n${lowered_ir}")
endif()
