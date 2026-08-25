cmake_minimum_required(VERSION 3.20)

project(TensorForge VERSION 0.1.0 LANGUAGES C CXX)

# LLVM 22's HandleLLVMOptions module uses CMake's pre-4.4 macro escape
# behavior. Select it explicitly so CMake 4.4 does not emit a policy warning.
if(POLICY CMP0219)
  cmake_policy(SET CMP0219 OLD)
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard used to build TensorForge")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_definitions(${LLVM_DEFINITIONS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_subdirectory(include/TensorForge/Dialect/TensorForge/IR)
add_subdirectory(lib/Dialect/TensorForge/IR)
add_subdirectory(lib/Transforms)
add_subdirectory(tools/tensorforge-opt)

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
