#include "CudaKernels.h"

#include <cuda_runtime.h>

namespace tensorforge::cuda {
namespace {

constexpr int kTileSize = 16;
constexpr int kElementwiseBlockSize = 256;

__device__ float geluTanh(float value) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kCubicCoefficient = 0.044715f;
  const float cubic = value * value * value;
  return 0.5f * value *
         (1.0f + tanhf(kSqrtTwoOverPi * (value + kCubicCoefficient * cubic)));
}

template <bool AddBias, bool ApplyGelu>
__global__ void
matmulF32Kernel(const float *__restrict__ lhs, const float *__restrict__ rhs,
                const float *__restrict__ bias, float *__restrict__ output,
                int64_t m, int64_t k, int64_t n) {
  __shared__ float lhsTile[kTileSize][kTileSize];
  __shared__ float rhsTile[kTileSize][kTileSize];

  const int64_t row =
      static_cast<int64_t>(blockIdx.y) * kTileSize + threadIdx.y;
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * kTileSize + threadIdx.x;
  float accumulator = 0.0f;

  for (int64_t tileStart = 0; tileStart < k; tileStart += kTileSize) {
    const int64_t lhsColumn = tileStart + threadIdx.x;
    const int64_t rhsRow = tileStart + threadIdx.y;

    lhsTile[threadIdx.y][threadIdx.x] =
        row < m && lhsColumn < k ? lhs[row * k + lhsColumn] : 0.0f;
    rhsTile[threadIdx.y][threadIdx.x] =
        rhsRow < k && column < n ? rhs[rhsRow * n + column] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int tileIndex = 0; tileIndex < kTileSize; ++tileIndex)
      accumulator +=
          lhsTile[threadIdx.y][tileIndex] * rhsTile[tileIndex][threadIdx.x];
    __syncthreads();
  }

  if (row < m && column < n) {
    if constexpr (AddBias)
      accumulator += bias[column];
    if constexpr (ApplyGelu)
      accumulator = geluTanh(accumulator);
    output[row * n + column] = accumulator;
  }
}

__global__ void geluF32Kernel(const float *__restrict__ input,
                              float *__restrict__ output, int64_t elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        static_cast<int64_t>(threadIdx.x);
  if (index < elements)
    output[index] = geluTanh(input[index]);
}

template <bool AddBias, bool ApplyGelu>
cudaError_t launchMatmulLike(const float *lhs, const float *rhs,
                             const float *bias, float *output, int64_t m,
                             int64_t k, int64_t n, cudaStream_t stream) {
  const dim3 block(kTileSize, kTileSize);
  const dim3 grid(static_cast<unsigned int>((n + kTileSize - 1) / kTileSize),
                  static_cast<unsigned int>((m + kTileSize - 1) / kTileSize));
  matmulF32Kernel<AddBias, ApplyGelu>
      <<<grid, block, 0, stream>>>(lhs, rhs, bias, output, m, k, n);
  return cudaGetLastError();
}

} // namespace

cudaError_t launchMatmulF32(const float *lhs, const float *rhs, float *output,
                            int64_t m, int64_t k, int64_t n,
                            cudaStream_t stream) {
  return launchMatmulLike<false, false>(lhs, rhs, nullptr, output, m, k, n,
                                        stream);
}

cudaError_t launchLinearF32(const float *input, const float *weight,
                            const float *bias, float *output, int64_t m,
                            int64_t k, int64_t n, cudaStream_t stream) {
  return launchMatmulLike<true, false>(input, weight, bias, output, m, k, n,
                                       stream);
}

cudaError_t launchFusedLinearGeluF32(const float *input, const float *weight,
                                     const float *bias, float *output,
                                     int64_t m, int64_t k, int64_t n,
                                     cudaStream_t stream) {
  return launchMatmulLike<true, true>(input, weight, bias, output, m, k, n,
                                      stream);
}

cudaError_t launchGeluF32(const float *input, float *output, int64_t elements,
                          cudaStream_t stream) {
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kElementwiseBlockSize - 1) / kElementwiseBlockSize);
  geluF32Kernel<<<blocks, kElementwiseBlockSize, 0, stream>>>(input, output,
                                                              elements);
  return cudaGetLastError();
}

} // namespace tensorforge::cuda
