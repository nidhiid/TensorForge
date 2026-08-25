set(TENSORFORGE_CUDA_RUNTIME_SOURCES CudaRuntimeStub.cpp)

if(TENSORFORGE_ENABLE_CUDA)
  set(TENSORFORGE_CUDA_RUNTIME_SOURCES
      CudaRuntime.cpp
      MatMul.cu)
endif()

add_library(TensorForgeCudaRuntime SHARED
  ${TENSORFORGE_CUDA_RUNTIME_SOURCES}
)

target_include_directories(TensorForgeCudaRuntime
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_definitions(TensorForgeCudaRuntime
  PRIVATE TENSORFORGE_CUDA_RUNTIME_EXPORTS
  PUBLIC TENSORFORGE_HAS_CUDA=$<BOOL:${TENSORFORGE_ENABLE_CUDA}>
)

set_target_properties(TensorForgeCudaRuntime PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(TENSORFORGE_ENABLE_CUDA)
  target_link_libraries(TensorForgeCudaRuntime PRIVATE CUDA::cudart)
  set_target_properties(TensorForgeCudaRuntime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()
