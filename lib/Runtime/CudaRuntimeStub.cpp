#include "TensorForge/Runtime/CudaRuntime.h"

namespace {

constexpr const char *kUnavailableMessage =
    "TensorForge was built without CUDA support";

} // namespace

extern "C" TfCudaStatus tfCudaGetDeviceCount(int32_t *count) {
  if (!count)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  *count = 0;
  return TF_CUDA_SUCCESS;
}

extern "C" TfCudaStatus tfCudaCreate(TfCudaContext **context, int32_t device) {
  (void)device;
  if (!context)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  *context = nullptr;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus tfCudaDestroy(TfCudaContext *context) {
  return context ? TF_CUDA_ERROR_UNAVAILABLE : TF_CUDA_ERROR_INVALID_ARGUMENT;
}

extern "C" TfCudaStatus tfCudaMalloc(TfCudaContext *context, void **pointer,
                                     size_t bytes) {
  (void)context;
  (void)bytes;
  if (pointer)
    *pointer = nullptr;
  return !pointer ? TF_CUDA_ERROR_INVALID_ARGUMENT : TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus tfCudaFree(TfCudaContext *context, void *pointer) {
  (void)context;
  return pointer ? TF_CUDA_ERROR_UNAVAILABLE : TF_CUDA_ERROR_INVALID_ARGUMENT;
}

extern "C" TfCudaStatus tfCudaCopyHostToDevice(TfCudaContext *context,
                                               void *deviceDestination,
                                               const void *hostSource,
                                               size_t bytes) {
  (void)context;
  (void)bytes;
  if (!deviceDestination || !hostSource)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus tfCudaCopyDeviceToHost(TfCudaContext *context,
                                               void *hostDestination,
                                               const void *deviceSource,
                                               size_t bytes) {
  (void)context;
  (void)bytes;
  if (!hostDestination || !deviceSource)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus tfCudaMatmulF32(TfCudaContext *context,
                                        const float *lhsDevice,
                                        const float *rhsDevice,
                                        float *outputDevice, int64_t m,
                                        int64_t k, int64_t n) {
  (void)context;
  if (!lhsDevice || !rhsDevice || !outputDevice || m <= 0 || k <= 0 || n <= 0)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus tfCudaSynchronize(TfCudaContext *context) {
  return context ? TF_CUDA_ERROR_UNAVAILABLE : TF_CUDA_ERROR_INVALID_ARGUMENT;
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
  (void)context;
  return kUnavailableMessage;
}
