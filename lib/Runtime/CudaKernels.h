#ifndef TENSORFORGE_RUNTIME_CUDAKERNELS_H
#define TENSORFORGE_RUNTIME_CUDAKERNELS_H

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace tensorforge::cuda {

cudaError_t launchMatmulF32(const float *lhs, const float *rhs, float *output,
                            int64_t m, int64_t k, int64_t n,
                            cudaStream_t stream);

} // namespace tensorforge::cuda

#endif // TENSORFORGE_RUNTIME_CUDAKERNELS_H
