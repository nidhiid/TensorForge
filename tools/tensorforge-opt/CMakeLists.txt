add_llvm_executable(tensorforge-opt
  tensorforge-opt.cpp
)

llvm_update_compile_flags(tensorforge-opt)

set_target_properties(tensorforge-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(tensorforge-opt PRIVATE
  MLIRFuncDialect
  MLIROptLib
  TensorForgeDialect
)
