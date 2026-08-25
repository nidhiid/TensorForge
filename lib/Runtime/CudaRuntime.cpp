#include "TensorForge/Runtime/CudaRuntime.h"

#include "CudaKernels.h"

#include <cuda_runtime_api.h>

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
  if (!context || !lhsDevice || !rhsDevice || !outputDevice || m <= 0 ||
      k <= 0 || n <= 0)
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
    context->lastError = "MatMul grid exceeds the selected device limits";
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  }

  cudaError_t error = tensorforge::cuda::launchMatmulF32(
      lhsDevice, rhsDevice, outputDevice, m, k, n, context->stream);
  if (error != cudaSuccess)
    return setError(context, TF_CUDA_ERROR_LAUNCH, "MatMul launch", error);
  return TF_CUDA_SUCCESS;
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
