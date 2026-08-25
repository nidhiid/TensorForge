#include "TensorForge/Runtime/CudaRuntime.h"

#include <cstdio>
#include <cstdlib>

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

extern "C" TfCudaStatus
tfCudaLinearF32(TfCudaContext *context, const float *inputDevice,
                const float *weightDevice, const float *biasDevice,
                float *outputDevice, int64_t m, int64_t k, int64_t n) {
  (void)context;
  if (!inputDevice || !weightDevice || !biasDevice || !outputDevice || m <= 0 ||
      k <= 0 || n <= 0)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus
tfCudaFusedLinearGeluF32(TfCudaContext *context, const float *inputDevice,
                         const float *weightDevice, const float *biasDevice,
                         float *outputDevice, int64_t m, int64_t k, int64_t n) {
  return tfCudaLinearF32(context, inputDevice, weightDevice, biasDevice,
                         outputDevice, m, k, n);
}

extern "C" TfCudaStatus tfCudaGeluF32(TfCudaContext *context,
                                      const float *inputDevice,
                                      float *outputDevice, int64_t elements) {
  (void)context;
  if (!inputDevice || !outputDevice || elements <= 0)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus
tfCudaTimeLinearGeluF32(TfCudaContext *context, const float *inputDevice,
                        const float *weightDevice, const float *biasDevice,
                        float *scratchDevice, float *outputDevice, int64_t m,
                        int64_t k, int64_t n, int32_t fused, int32_t iterations,
                        float *millisecondsPerIteration) {
  (void)context;
  (void)fused;
  if (!inputDevice || !weightDevice || !biasDevice || !scratchDevice ||
      !outputDevice || m <= 0 || k <= 0 || n <= 0 || iterations <= 0 ||
      !millisecondsPerIteration)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  *millisecondsPerIteration = 0.0f;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus tfCudaRunMatmulHostF32(uintptr_t lhsHost,
                                               uintptr_t rhsHost,
                                               uintptr_t outputHost, int64_t m,
                                               int64_t k, int64_t n) {
  if (!lhsHost || !rhsHost || !outputHost || m <= 0 || k <= 0 || n <= 0)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus tfCudaRunLinearHostF32(uintptr_t inputHost,
                                               uintptr_t weightHost,
                                               uintptr_t biasHost,
                                               uintptr_t outputHost, int64_t m,
                                               int64_t k, int64_t n) {
  if (!inputHost || !weightHost || !biasHost || !outputHost || m <= 0 ||
      k <= 0 || n <= 0)
    return TF_CUDA_ERROR_INVALID_ARGUMENT;
  return TF_CUDA_ERROR_UNAVAILABLE;
}

extern "C" TfCudaStatus
tfCudaRunFusedLinearGeluHostF32(uintptr_t inputHost, uintptr_t weightHost,
                                uintptr_t biasHost, uintptr_t outputHost,
                                int64_t m, int64_t k, int64_t n) {
  return tfCudaRunLinearHostF32(inputHost, weightHost, biasHost, outputHost, m,
                                k, n);
}

extern "C" void tfCudaCheckStatus(int32_t status) {
  if (status == TF_CUDA_SUCCESS)
    return;
  std::fprintf(stderr, "TensorForge CUDA failure: %s (%s)\n",
               tfCudaStatusString(static_cast<TfCudaStatus>(status)),
               kUnavailableMessage);
  std::abort();
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
