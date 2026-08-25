#include "TensorForge/Runtime/CudaRuntime.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool check(TfCudaContext *context, TfCudaStatus status, const char *operation) {
  if (status == TF_CUDA_SUCCESS)
    return true;
  std::cerr << operation << " failed: " << tfCudaStatusString(status) << " ("
            << tfCudaGetLastError(context) << ")\n";
  return false;
}

std::vector<float> referenceMatmul(const std::vector<float> &lhs,
                                   const std::vector<float> &rhs, int64_t m,
                                   int64_t k, int64_t n) {
  std::vector<float> output(static_cast<size_t>(m * n), 0.0f);
  for (int64_t row = 0; row < m; ++row)
    for (int64_t column = 0; column < n; ++column)
      for (int64_t inner = 0; inner < k; ++inner)
        output[static_cast<size_t>(row * n + column)] +=
            lhs[static_cast<size_t>(row * k + inner)] *
            rhs[static_cast<size_t>(inner * n + column)];
  return output;
}

bool runCase(TfCudaContext *context, int64_t m, int64_t k, int64_t n) {
  std::vector<float> lhs(static_cast<size_t>(m * k));
  std::vector<float> rhs(static_cast<size_t>(k * n));
  for (size_t index = 0; index < lhs.size(); ++index)
    lhs[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 7.0f;
  for (size_t index = 0; index < rhs.size(); ++index)
    rhs[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 9.0f;

  const std::vector<float> expected = referenceMatmul(lhs, rhs, m, k, n);
  std::vector<float> actual(expected.size(), 0.0f);
  void *lhsDevice = nullptr;
  void *rhsDevice = nullptr;
  void *outputDevice = nullptr;

  const size_t lhsBytes = lhs.size() * sizeof(float);
  const size_t rhsBytes = rhs.size() * sizeof(float);
  const size_t outputBytes = actual.size() * sizeof(float);

  bool succeeded =
      check(context, tfCudaMalloc(context, &lhsDevice, lhsBytes),
            "lhs allocation") &&
      check(context, tfCudaMalloc(context, &rhsDevice, rhsBytes),
            "rhs allocation") &&
      check(context, tfCudaMalloc(context, &outputDevice, outputBytes),
            "output allocation") &&
      check(context,
            tfCudaCopyHostToDevice(context, lhsDevice, lhs.data(), lhsBytes),
            "lhs upload") &&
      check(context,
            tfCudaCopyHostToDevice(context, rhsDevice, rhs.data(), rhsBytes),
            "rhs upload") &&
      check(context,
            tfCudaMatmulF32(context, static_cast<const float *>(lhsDevice),
                            static_cast<const float *>(rhsDevice),
                            static_cast<float *>(outputDevice), m, k, n),
            "MatMul launch") &&
      check(context,
            tfCudaCopyDeviceToHost(context, actual.data(), outputDevice,
                                   outputBytes),
            "output download") &&
      check(context, tfCudaSynchronize(context), "stream synchronization");

  if (succeeded) {
    for (size_t index = 0; index < actual.size(); ++index) {
      const float tolerance =
          2.0e-4f * std::max(1.0f, std::abs(expected[index]));
      if (std::abs(actual[index] - expected[index]) > tolerance) {
        std::cerr << "MatMul mismatch for [" << m << ',' << k << "] x [" << k
                  << ',' << n << "] at element " << index
                  << ": expected=" << expected[index]
                  << " actual=" << actual[index] << '\n';
        succeeded = false;
        break;
      }
    }
  }

  if (outputDevice)
    succeeded =
        check(context, tfCudaFree(context, outputDevice), "output free") &&
        succeeded;
  if (rhsDevice)
    succeeded =
        check(context, tfCudaFree(context, rhsDevice), "rhs free") && succeeded;
  if (lhsDevice)
    succeeded =
        check(context, tfCudaFree(context, lhsDevice), "lhs free") && succeeded;
  return succeeded;
}

} // namespace

int main() {
  int32_t deviceCount = 0;
  TfCudaStatus countStatus = tfCudaGetDeviceCount(&deviceCount);
  if (countStatus != TF_CUDA_SUCCESS || deviceCount == 0) {
    std::cout << "No CUDA device is available; skipping GPU correctness test\n";
    return 77;
  }

  TfCudaContext *context = nullptr;
  if (!check(context, tfCudaCreate(&context, 0), "context creation"))
    return 1;

  // The second and third cases exercise all boundary checks because none of
  // their dimensions is a multiple of the 16x16 kernel tile.
  bool succeeded = runCase(context, 2, 3, 4) && runCase(context, 17, 19, 13) &&
                   runCase(context, 31, 7, 29);
  succeeded = check(context, tfCudaDestroy(context), "context destruction") &&
              succeeded;
  return succeeded ? 0 : 1;
}
