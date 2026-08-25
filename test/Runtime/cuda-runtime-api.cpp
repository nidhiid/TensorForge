#include "TensorForge/Runtime/CudaRuntime.h"

#include <cstring>
#include <iostream>

int main() {
  if (tfCudaGetDeviceCount(nullptr) != TF_CUDA_ERROR_INVALID_ARGUMENT) {
    std::cerr << "device-count query accepted a null output pointer\n";
    return 1;
  }

  int32_t deviceCount = -1;
  TfCudaStatus countStatus = tfCudaGetDeviceCount(&deviceCount);
  if (countStatus != TF_CUDA_SUCCESS || deviceCount < 0) {
    std::cerr << "device-count query failed: "
              << tfCudaStatusString(countStatus) << '\n';
    return 1;
  }

  if (std::strcmp(tfCudaStatusString(TF_CUDA_ERROR_LAUNCH),
                  "CUDA kernel launch failed") != 0) {
    std::cerr << "status strings are not stable\n";
    return 1;
  }

#if !TENSORFORGE_HAS_CUDA
  if (deviceCount != 0) {
    std::cerr << "non-CUDA build reported a CUDA device\n";
    return 1;
  }

  TfCudaContext *context = reinterpret_cast<TfCudaContext *>(0x1);
  TfCudaStatus createStatus = tfCudaCreate(&context, 0);
  if (createStatus != TF_CUDA_ERROR_UNAVAILABLE || context != nullptr) {
    std::cerr << "non-CUDA context creation did not report unavailable\n";
    return 1;
  }
  if (std::strstr(tfCudaGetLastError(nullptr), "without CUDA") == nullptr) {
    std::cerr << "non-CUDA build did not provide a useful diagnostic\n";
    return 1;
  }
#endif

  std::cout << "TensorForge CUDA runtime API is available; devices="
            << deviceCount << '\n';
  return 0;
}
