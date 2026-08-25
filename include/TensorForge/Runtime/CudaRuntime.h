#ifndef TENSORFORGE_RUNTIME_CUDARUNTIME_H
#define TENSORFORGE_RUNTIME_CUDARUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TENSORFORGE_CUDA_RUNTIME_EXPORTS)
#define TF_CUDA_API __declspec(dllexport)
#else
#define TF_CUDA_API __declspec(dllimport)
#endif
#else
#define TF_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum TfCudaStatus {
  TF_CUDA_SUCCESS = 0,
  TF_CUDA_ERROR_INVALID_ARGUMENT = 1,
  TF_CUDA_ERROR_UNAVAILABLE = 2,
  TF_CUDA_ERROR_ALLOCATION = 3,
  TF_CUDA_ERROR_COPY = 4,
  TF_CUDA_ERROR_LAUNCH = 5,
  TF_CUDA_ERROR_SYNCHRONIZATION = 6,
  TF_CUDA_ERROR_INTERNAL = 7
} TfCudaStatus;

typedef struct TfCudaContext TfCudaContext;

/// Report the number of usable CUDA devices. A non-CUDA build reports zero.
TF_CUDA_API TfCudaStatus tfCudaGetDeviceCount(int32_t *count);

/// Create one runtime context with one non-blocking stream on `device`.
TF_CUDA_API TfCudaStatus tfCudaCreate(TfCudaContext **context, int32_t device);
TF_CUDA_API TfCudaStatus tfCudaDestroy(TfCudaContext *context);

/// Allocate and free memory on the context's device.
TF_CUDA_API TfCudaStatus tfCudaMalloc(TfCudaContext *context, void **pointer,
                                      size_t bytes);
TF_CUDA_API TfCudaStatus tfCudaFree(TfCudaContext *context, void *pointer);

/// Queue copies on the context's stream.
TF_CUDA_API TfCudaStatus tfCudaCopyHostToDevice(TfCudaContext *context,
                                                void *deviceDestination,
                                                const void *hostSource,
                                                size_t bytes);
TF_CUDA_API TfCudaStatus tfCudaCopyDeviceToHost(TfCudaContext *context,
                                                void *hostDestination,
                                                const void *deviceSource,
                                                size_t bytes);

/// Queue row-major FP32 `[M,K] x [K,N] -> [M,N]` MatMul.
TF_CUDA_API TfCudaStatus tfCudaMatmulF32(TfCudaContext *context,
                                         const float *lhsDevice,
                                         const float *rhsDevice,
                                         float *outputDevice, int64_t m,
                                         int64_t k, int64_t n);

/// Wait for every copy and kernel queued on the context's stream.
TF_CUDA_API TfCudaStatus tfCudaSynchronize(TfCudaContext *context);

/// Return a stable status name or the context's most recent detailed error.
TF_CUDA_API const char *tfCudaStatusString(TfCudaStatus status);
TF_CUDA_API const char *tfCudaGetLastError(const TfCudaContext *context);

#ifdef __cplusplus
}
#endif

#endif // TENSORFORGE_RUNTIME_CUDARUNTIME_H
