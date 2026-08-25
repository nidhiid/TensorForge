#include "CudaKernels.h"

#include <cuda_runtime.h>

namespace tensorforge::cuda {
namespace {

constexpr int kTileSize = 16;

__global__ void matmulF32Kernel(const float *__restrict__ lhs,
                                const float *__restrict__ rhs,
                                float *__restrict__ output, int64_t m,
                                int64_t k, int64_t n) {
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

  if (row < m && column < n)
    output[row * n + column] = accumulator;
}

} // namespace

cudaError_t launchMatmulF32(const float *lhs, const float *rhs, float *output,
                            int64_t m, int64_t k, int64_t n,
                            cudaStream_t stream) {
  const dim3 block(kTileSize, kTileSize);
  const dim3 grid(static_cast<unsigned int>((n + kTileSize - 1) / kTileSize),
                  static_cast<unsigned int>((m + kTileSize - 1) / kTileSize));
  matmulF32Kernel<<<grid, block, 0, stream>>>(lhs, rhs, output, m, k, n);
  return cudaGetLastError();
}

} // namespace tensorforge::cuda
