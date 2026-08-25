execute_process(
  COMMAND "${TENSORFORGE_OPT}" --tf-fuse-linear-gelu "${FUSION_INPUT}"
  RESULT_VARIABLE fusion_result
  OUTPUT_VARIABLE fusion_output
  ERROR_VARIABLE fusion_error
)
if(NOT fusion_result EQUAL 0)
  message(FATAL_ERROR "Fusion command failed:\n${fusion_error}")
endif()

string(FIND "${fusion_output}" "tf.fused_linear_gelu" fused_position)
string(FIND "${fusion_output}" "tf.linear" linear_position)
string(FIND "${fusion_output}" "tf.gelu" gelu_position)
if(fused_position EQUAL -1 OR NOT linear_position EQUAL -1 OR
   NOT gelu_position EQUAL -1)
  message(FATAL_ERROR "Expected only the fused operation:\n${fusion_output}")
endif()

execute_process(
  COMMAND "${TENSORFORGE_OPT}" --tf-fuse-linear-gelu "${NO_FUSION_INPUT}"
  RESULT_VARIABLE no_fusion_result
  OUTPUT_VARIABLE no_fusion_output
  ERROR_VARIABLE no_fusion_error
)
if(NOT no_fusion_result EQUAL 0)
  message(FATAL_ERROR "No-fusion command failed:\n${no_fusion_error}")
endif()

string(FIND "${no_fusion_output}" "tf.fused_linear_gelu" fused_position)
string(FIND "${no_fusion_output}" "tf.linear" linear_position)
string(FIND "${no_fusion_output}" "tf.gelu" gelu_position)
if(NOT fused_position EQUAL -1 OR linear_position EQUAL -1 OR
   gelu_position EQUAL -1)
  message(FATAL_ERROR
          "Multiple-user Linear should remain unfused:\n${no_fusion_output}")
endif()

execute_process(
  COMMAND "${TENSORFORGE_OPT}" --canonicalize "${CANONICALIZE_INPUT}"
  RESULT_VARIABLE canonicalize_result
  OUTPUT_VARIABLE canonicalize_output
  ERROR_VARIABLE canonicalize_error
)
if(NOT canonicalize_result EQUAL 0)
  message(FATAL_ERROR "Canonicalize command failed:\n${canonicalize_error}")
endif()

string(FIND "${canonicalize_output}" "tf.reshape" reshape_position)
string(FIND "${canonicalize_output}" "tf.transpose" transpose_position)
if(NOT reshape_position EQUAL -1 OR NOT transpose_position EQUAL -1)
  message(FATAL_ERROR
          "Expected reshape and transpose chains to disappear:\n${canonicalize_output}")
endif()
