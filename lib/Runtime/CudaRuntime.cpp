#include "TensorForge/Runtime/CudaRuntime.h"

#include "CudaKernels.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>

struct TfCudaContext {
  int32_t device = 0;
  cudaStream_t stream = nullptr;
  cudaDeviceProp properties{};
  std::string lastError;
};

namespace {

thread_local std::string globalError = "no CUDA error";

TfCudaStatus setError(TfCudaContext *context, TfCudaStatus status,
                      const char *operation, cudaError_t error) {
  std::string message = operation;
  message += ": ";
  message += cudaGetErrorString(error);
  if (context)
    context->lastError = message;
  else
    globalError = message;
  return status;
}

TfCudaStatus selectDevice(TfCudaContext *context) {
  if (!context)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  cudaError_t error = cudaSetDevice(context->device);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_UNAVAILABLE, "cudaSetDevice", error);
  return TF_CUDA_SUCCESS;
}

bool invalidCopyArguments(TfCudaContext *context, const void *destination,
                          const void *source, size_t bytes) {
  return !context || !destination || !source || bytes == 0;
}

TfCudaStatus validateMatmulLaunch(TfCudaContext *context, const float *lhs,
                                  const float *rhs, const float *bias,
                                  float *output, int64_t m, int64_t k,
                                  int64_t n, bool requiresBias) {
  if (!context || !lhs || !rhs || !output || (requiresBias && !bias) ||
      m <= 0 || k <= 0 || n <= 0)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  if (m > std::numeric_limits<int64_t>::max() - 15 ||
      k > std::numeric_limits<int64_t>::max() - 15 ||
      n > std::numeric_limits<int64_t>::max() - 15)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  if (TfCudaStatus status = selectDevice(context); status != TF_CUDA_SUCCESS)
    return status;

  const int64_t gridX = (n + 15) / 16;
  const int64_t gridY = (m + 15) / 16;
  if (gridX > context->properties.maxGridSize[0] ||
      gridY > context->properties.maxGridSize[1]) {
    context->lastError = "matrix grid exceeds the selected device limits";
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  }
  return TF_CUDA_SUCCESS;
}

bool matrixBytes(int64_t rows, int64_t columns, size_t &bytes) {
  if (rows <= 0 || columns <= 0)
    return false;
  const uint64_t unsignedRows = static_cast<uint64_t>(rows);
  const uint64_t unsignedColumns = static_cast<uint64_t>(columns);
  if (unsignedRows > std::numeric_limits<size_t>::max() / unsignedColumns)
    return false;
  const size_t elements = static_cast<size_t>(unsignedRows * unsignedColumns);
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float))
    return false;
  bytes = elements * sizeof(float);
  return true;
}

enum class HostOperation { Matmul, Linear, FusedLinearGelu };

TfCudaStatus runHostOperation(HostOperation operation, uintptr_t inputHost,
                              uintptr_t weightHost, uintptr_t biasHost,
                              uintptr_t outputHost, int64_t m, int64_t k,
                              int64_t n) {
  if (!inputHost || !weightHost || !outputHost ||
      (operation != HostOperation::Matmul && !biasHost))
    return TF_CUDA_ERROR_INVALID_ARGUMENT;

  size_t inputBytes = 0;
  size_t weightBytes = 0;
  size_t outputBytes = 0;
  size_t biasBytes = 0;
  if (!matrixBytes(m, k, inputBytes) || !matrixBytes(k, n, weightBytes) ||
      !matrixBytes(m, n, outputBytes) ||
      (operation != HostOperation::Matmul && !matrixBytes(1, n, biasBytes)))
    return TF_CUDA_ERROR_INVALID_ARGUMENT;

  TfCudaContext *context = nullptr;
  TfCudaStatus status = tfCudaCreate(&context, 0);
  if (status != TF_CUDA_SUCCESS)
    return status;

  void *inputDevice = nullptr;
  void *weightDevice = nullptr;
  void *biasDevice = nullptr;
  void *outputDevice = nullptr;
  auto runIfSuccessful = [&](auto &&operationToRun) {
    if (status == TF_CUDA_SUCCESS)
      status = operationToRun();
  };

  runIfSuccessful(
      [&] { return tfCudaMalloc(context, &inputDevice, inputBytes); });
  runIfSuccessful(
      [&] { return tfCudaMalloc(context, &weightDevice, weightBytes); });
  if (operation != HostOperation::Matmul)
    runIfSuccessful(
        [&] { return tfCudaMalloc(context, &biasDevice, biasBytes); });
  runIfSuccessful(
      [&] { return tfCudaMalloc(context, &outputDevice, outputBytes); });
  runIfSuccessful([&] {
    return tfCudaCopyHostToDevice(context, inputDevice,
                                  reinterpret_cast<const void *>(inputHost),
                                  inputBytes);
  });
  runIfSuccessful([&] {
    return tfCudaCopyHostToDevice(context, weightDevice,
                                  reinterpret_cast<const void *>(weightHost),
                                  weightBytes);
  });
  if (operation != HostOperation::Matmul)
    runIfSuccessful([&] {
      return tfCudaCopyHostToDevice(context, biasDevice,
                                    reinterpret_cast<const void *>(biasHost),
                                    biasBytes);
    });
  runIfSuccessful([&] {
    if (operation == HostOperation::Matmul)
      return tfCudaMatmulF32(context, static_cast<const float *>(inputDevice),
                             static_cast<const float *>(weightDevice),
                             static_cast<float *>(outputDevice), m, k, n);
    if (operation == HostOperation::Linear)
      return tfCudaLinearF32(context, static_cast<const float *>(inputDevice),
                             static_cast<const float *>(weightDevice),
                             static_cast<const float *>(biasDevice),
                             static_cast<float *>(outputDevice), m, k, n);
    return tfCudaFusedLinearGeluF32(
        context, static_cast<const float *>(inputDevice),
        static_cast<const float *>(weightDevice),
        static_cast<const float *>(biasDevice),
        static_cast<float *>(outputDevice), m, k, n);
  });
  runIfSuccessful([&] {
    return tfCudaCopyDeviceToHost(context, reinterpret_cast<void *>(outputHost),
                                  outputDevice, outputBytes);
  });
  runIfSuccessful([&] { return tfCudaSynchronize(context); });

  auto freeBuffer = [&](void *buffer) {
    if (!buffer)
      return;
    TfCudaStatus freeStatus = tfCudaFree(context, buffer);
    if (status == TF_CUDA_SUCCESS)
      status = freeStatus;
  };
  freeBuffer(outputDevice);
  freeBuffer(biasDevice);
  freeBuffer(weightDevice);
  freeBuffer(inputDevice);
  std::string operationError = context->lastError;
  TfCudaStatus destroyStatus = tfCudaDestroy(context);
  if (status == TF_CUDA_SUCCESS)
    status = destroyStatus;
  if (status != TF_CUDA_SUCCESS && !operationError.empty())
    globalError = operationError;
  return status;
}

} // namespace

extern "C" TfCudaStatus tfCudaGetDeviceCount(int32_t *count) {
  if (!count)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;

  int runtimeCount = 0;
  cudaError_t error = cudaGetDeviceCount(&runtimeCount);
  if (error == cudaErrorNoDevice) {
    (void)cudaGetLastError();
    *count = 0;
    return TF_CUDA_SUCCESS;
  }
  if (error != cudaSuccess) {
    *count = 0;
    return setError(nullptr, TF_CUDA_ERROR_UNAVAILABLE, "cudaGetDeviceCount",
                    error);
  }

  *count = runtimeCount;
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaCreate(TfCudaContext **context, int32_t device) {
  if (!context || device < 0)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  *context = nullptr;

  int32_t count = 0;
  TfCudaStatus countStatus = tfCudaGetDeviceCount(&count);
  if (countStatus != TF_CUDA_SUCCESS)
    return TF_CUDA_ERROR_UNAVAILABLE;
  if (device >= count) {
    globalError = "requested CUDA device is unavailable";
    return TF_CUDA_ERROR_UNAVAILABLE;
  }

  auto *created = new (std::nothrow) TfCudaContext();
  if (!created) {
    globalError = "failed to allocate the CUDA runtime context";
    return TF_CUDA_ERROR_ALLOCATION;
  }
  created->device = device;

  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) {
    delete created;
    return setError(nullptr, TF_CUDA_ERROR_UNAVAILABLE, "cudaSetDevice", error);
  }

  error = cudaGetDeviceProperties(&created->properties, device);
  if (error != cudaSuccess) {
    delete created;
    return setError(nullptr, TF_CUDA_ERROR_UNAVAILABLE,
                    "cudaGetDeviceProperties", error);
  }

  error = cudaStreamCreateWithFlags(&created->stream, cudaStreamNonBlocking);
  if (error != cudaSuccess) {
    delete created;
    return setError(nullptr, TF_CUDA_ERROR_INTERNAL,
                    "cudaStreamCreateWithFlags", error);
  }

  *context = created;
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaDestroy(TfCudaContext *context) {
  if (!context)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;

  cudaError_t selectError = cudaSetDevice(context->device);
  cudaError_t destroyError = selectError == cudaSuccess
                                 ? cudaStreamDestroy(context->stream)
                                 : selectError;
  delete context;
  return destroyError == cudaSuccess ? TF_CUDA_SUCCESS : TF_CUDA_ERROR_INTERNAL;
}

extern "C" TfCudaStatus tfCudaMalloc(TfCudaContext *context, void **pointer,
                                     size_t bytes) {
  if (!context || !pointer || bytes == 0)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  *pointer = nullptr;
  if (TfCudaStatus status = selectDevice(context); status != TF_CUDA_SUCCESS)
    return status;

  cudaError_t error = cudaMalloc(pointer, bytes);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_ALLOCATION, "cudaMalloc", error);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaFree(TfCudaContext *context, void *pointer) {
  if (!context || !pointer)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  if (TfCudaStatus status = selectDevice(context); status != TF_CUDA_SUCCESS)
    return status;

  cudaError_t error = cudaFree(pointer);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_INTERNAL, "cudaFree", error);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaCopyHostToDevice(TfCudaContext *context,
                                               void *deviceDestination,
                                               const void *hostSource,
                                               size_t bytes) {
  if (invalidCopyArguments(context, deviceDestination, hostSource, bytes))
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  if (TfCudaStatus status = selectDevice(context); status != TF_CUDA_SUCCESS)
    return status;

  cudaError_t error = cudaMemcpyAsync(deviceDestination, hostSource, bytes,
                                      cudaMemcpyHostToDevice, context->stream);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_COPY, "host-to-device copy", error);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaCopyDeviceToHost(TfCudaContext *context,
                                               void *hostDestination,
                                               const void *deviceSource,
                                               size_t bytes) {
  if (invalidCopyArguments(context, hostDestination, deviceSource, bytes))
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  if (TfCudaStatus status = selectDevice(context); status != TF_CUDA_SUCCESS)
    return status;

  cudaError_t error = cudaMemcpyAsync(hostDestination, deviceSource, bytes,
                                      cudaMemcpyDeviceToHost, context->stream);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_COPY, "device-to-host copy", error);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaMatmulF32(TfCudaContext *context,
                                        const float *lhsDevice,
                                        const float *rhsDevice,
                                        float *outputDevice, int64_t m,
                                        int64_t k, int64_t n) {
  TfCudaStatus validation = validateMatmulLaunch(
      context, lhsDevice, rhsDevice, nullptr, outputDevice, m, k, n, false);
  if (validation != TF_CUDA_SUCCESS)
    return validation;

  cudaError_t error = tensorforge::cuda::launchMatmulF32(
      lhsDevice, rhsDevice, outputDevice, m, k, n, context->stream);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_LAUNCH, "MatMul launch", error);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus
tfCudaLinearF32(TfCudaContext *context, const float *inputDevice,
                const float *weightDevice, const float *biasDevice,
                float *outputDevice, int64_t m, int64_t k, int64_t n) {
  TfCudaStatus validation =
      validateMatmulLaunch(context, inputDevice, weightDevice, biasDevice,
                           outputDevice, m, k, n, true);
  if (validation != TF_CUDA_SUCCESS)
    return validation;

  cudaError_t error = tensorforge::cuda::launchLinearF32(
      inputDevice, weightDevice, biasDevice, outputDevice, m, k, n,
      context->stream);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_LAUNCH, "Linear launch", error);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus
tfCudaFusedLinearGeluF32(TfCudaContext *context, const float *inputDevice,
                         const float *weightDevice, const float *biasDevice,
                         float *outputDevice, int64_t m, int64_t k, int64_t n) {
  TfCudaStatus validation =
      validateMatmulLaunch(context, inputDevice, weightDevice, biasDevice,
                           outputDevice, m, k, n, true);
  if (validation != TF_CUDA_SUCCESS)
    return validation;

  cudaError_t error = tensorforge::cuda::launchFusedLinearGeluF32(
      inputDevice, weightDevice, biasDevice, outputDevice, m, k, n,
      context->stream);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_LAUNCH, "fused Linear+GELU launch",
                    error);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaGeluF32(TfCudaContext *context,
                                      const float *inputDevice,
                                      float *outputDevice, int64_t elements) {
  if (!context || !inputDevice || !outputDevice || elements <= 0 ||
      elements > std::numeric_limits<int64_t>::max() - 255)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  if (TfCudaStatus status = selectDevice(context); status != TF_CUDA_SUCCESS)
    return status;

  const int64_t blocks = (elements + 255) / 256;
  if (blocks > context->properties.maxGridSize[0]) {
    context->lastError = "GELU grid exceeds the selected device limits";
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  }
  cudaError_t error = tensorforge::cuda::launchGeluF32(
      inputDevice, outputDevice, elements, context->stream);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_LAUNCH, "GELU launch", error);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus
tfCudaTimeLinearGeluF32(TfCudaContext *context, const float *inputDevice,
                        const float *weightDevice, const float *biasDevice,
                        float *scratchDevice, float *outputDevice, int64_t m,
                        int64_t k, int64_t n, int32_t fused, int32_t iterations,
                        float *millisecondsPerIteration) {
  if (!scratchDevice || !millisecondsPerIteration || m <= 0 || k <= 0 ||
      n <= 0 || iterations <= 0 || m > std::numeric_limits<int64_t>::max() / n)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  *millisecondsPerIteration = 0.0f;
  TfCudaStatus validation =
      validateMatmulLaunch(context, inputDevice, weightDevice, biasDevice,
                           outputDevice, m, k, n, true);
  if (validation != TF_CUDA_SUCCESS)
    return validation;

  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  cudaError_t error = cudaEventCreate(&start);
  if (error == cudaSuccess)
    error = cudaEventCreate(&end);
  if (error == cudaSuccess)
    error = cudaEventRecord(start, context->stream);

  for (int32_t iteration = 0; iteration < iterations && error == cudaSuccess;
       ++iteration) {
    if (fused) {
      error = tensorforge::cuda::launchFusedLinearGeluF32(
          inputDevice, weightDevice, biasDevice, outputDevice, m, k, n,
          context->stream);
    } else {
      error = tensorforge::cuda::launchLinearF32(inputDevice, weightDevice,
                                                 biasDevice, scratchDevice, m,
                                                 k, n, context->stream);
      if (error == cudaSuccess)
        error = tensorforge::cuda::launchGeluF32(scratchDevice, outputDevice,
                                                 m * n, context->stream);
    }
  }

  if (error == cudaSuccess)
    error = cudaEventRecord(end, context->stream);
  if (error == cudaSuccess)
    error = cudaEventSynchronize(end);
  float totalMilliseconds = 0.0f;
  if (error == cudaSuccess)
    error = cudaEventElapsedTime(&totalMilliseconds, start, end);

  if (end)
    (void)cudaEventDestroy(end);
  if (start)
    (void)cudaEventDestroy(start);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_SYNCHRONIZATION,
                    "Linear+GELU timing", error);

  *millisecondsPerIteration =
      totalMilliseconds / static_cast<float>(iterations);
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaRunMatmulHostF32(uintptr_t lhsHost,
                                               uintptr_t rhsHost,
                                               uintptr_t outputHost, int64_t m,
                                               int64_t k, int64_t n) {
  return runHostOperation(HostOperation::Matmul, lhsHost, rhsHost, 0,
                          outputHost, m, k, n);
}

extern "C" TfCudaStatus tfCudaRunLinearHostF32(uintptr_t inputHost,
                                               uintptr_t weightHost,
                                               uintptr_t biasHost,
                                               uintptr_t outputHost, int64_t m,
                                               int64_t k, int64_t n) {
  return runHostOperation(HostOperation::Linear, inputHost, weightHost,
                          biasHost, outputHost, m, k, n);
}

extern "C" TfCudaStatus
tfCudaRunFusedLinearGeluHostF32(uintptr_t inputHost, uintptr_t weightHost,
                                uintptr_t biasHost, uintptr_t outputHost,
                                int64_t m, int64_t k, int64_t n) {
  return runHostOperation(HostOperation::FusedLinearGelu, inputHost, weightHost,
                          biasHost, outputHost, m, k, n);
}

extern "C" void tfCudaCheckStatus(int32_t status) {
  if (status == TF_CUDA_SUCCESS)
    return;
  std::fprintf(stderr, "TensorForge CUDA failure: %s (%s)\n",
               tfCudaStatusString(static_cast<TfCudaStatus>(status)),
               tfCudaGetLastError(nullptr));
  std::abort();
}

extern "C" TfCudaStatus tfCudaSynchronize(TfCudaContext *context) {
  if (!context)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  if (TfCudaStatus status = selectDevice(context); status != TF_CUDA_SUCCESS)
    return status;

  cudaError_t error = cudaStreamSynchronize(context->stream);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_SYNCHRONIZATION,
                    "cudaStreamSynchronize", error);
  return TF_CUDA_SUCCESS;
}

extern "C" const char *tfCudaStatusString(TfCudaStatus status) {
  switch (status) {
  case TF_CUDA_SUCCESS:
    return "success";
  case TF_CUDA_ERROR_INVALID_ARGUMENT:
    return "invalid argument";
  case TF_CUDA_ERROR_UNAVAILABLE:
    return "CUDA unavailable";
  case TF_CUDA_ERROR_ALLOCATION:
    return "CUDA allocation failed";
  case TF_CUDA_ERROR_COPY:
    return "CUDA copy failed";
  case TF_CUDA_ERROR_LAUNCH:
    return "CUDA kernel launch failed";
  case TF_CUDA_ERROR_SYNCHRONIZATION:
    return "CUDA synchronization failed";
  case TF_CUDA_ERROR_INTERNAL:
    return "CUDA internal error";
  }
  return "unknown CUDA status";
}

extern "C" const char *tfCudaGetLastError(const TfCudaContext *context) {
  if (!context)
    return globalError.c_str();
  return context->lastError.empty() ? "no CUDA error"
                                    : context->lastError.c_str();
}
