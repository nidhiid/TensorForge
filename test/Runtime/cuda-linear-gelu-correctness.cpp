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

float gelu(float value) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kCubicCoefficient = 0.044715f;
  const float cubic = value * value * value;
  return 0.5f * value *
         (1.0f +
          std::tanh(kSqrtTwoOverPi * (value + kCubicCoefficient * cubic)));
}

} // namespace

int main() {
  int32_t deviceCount = 0;
  if (tfCudaGetDeviceCount(&deviceCount) != TF_CUDA_SUCCESS ||
      deviceCount == 0) {
    std::cout << "No CUDA device is available; skipping GPU correctness test\n";
    return 77;
  }

  constexpr int64_t m = 17;
  constexpr int64_t k = 19;
  constexpr int64_t n = 13;
  std::vector<float> input(static_cast<size_t>(m * k));
  std::vector<float> weight(static_cast<size_t>(k * n));
  std::vector<float> bias(static_cast<size_t>(n));
  for (size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<float>(static_cast<int>(index % 9) - 4) / 8.0f;
  for (size_t index = 0; index < weight.size(); ++index)
    weight[index] = static_cast<float>(static_cast<int>(index % 7) - 3) / 10.0f;
  for (size_t index = 0; index < bias.size(); ++index)
    bias[index] = static_cast<float>(static_cast<int>(index) - 6) / 20.0f;

  std::vector<float> expected(static_cast<size_t>(m * n), 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t column = 0; column < n; ++column) {
      float value = bias[static_cast<size_t>(column)];
      for (int64_t inner = 0; inner < k; ++inner)
        value += input[static_cast<size_t>(row * k + inner)] *
                 weight[static_cast<size_t>(inner * n + column)];
      expected[static_cast<size_t>(row * n + column)] = gelu(value);
    }
  }

  TfCudaContext *context = nullptr;
  if (!check(context, tfCudaCreate(&context, 0), "context creation"))
    return 1;

  void *inputDevice = nullptr;
  void *weightDevice = nullptr;
  void *biasDevice = nullptr;
  void *scratchDevice = nullptr;
  void *fusedDevice = nullptr;
  void *unfusedDevice = nullptr;
  const size_t inputBytes = input.size() * sizeof(float);
  const size_t weightBytes = weight.size() * sizeof(float);
  const size_t biasBytes = bias.size() * sizeof(float);
  const size_t outputBytes = expected.size() * sizeof(float);

  bool succeeded =
      check(context, tfCudaMalloc(context, &inputDevice, inputBytes),
            "input allocation") &&
      check(context, tfCudaMalloc(context, &weightDevice, weightBytes),
            "weight allocation") &&
      check(context, tfCudaMalloc(context, &biasDevice, biasBytes),
            "bias allocation") &&
      check(context, tfCudaMalloc(context, &scratchDevice, outputBytes),
            "scratch allocation") &&
      check(context, tfCudaMalloc(context, &fusedDevice, outputBytes),
            "fused output allocation") &&
      check(context, tfCudaMalloc(context, &unfusedDevice, outputBytes),
            "unfused output allocation") &&
      check(context,
            tfCudaCopyHostToDevice(context, inputDevice, input.data(),
                                   inputBytes),
            "input upload") &&
      check(context,
            tfCudaCopyHostToDevice(context, weightDevice, weight.data(),
                                   weightBytes),
            "weight upload") &&
      check(context,
            tfCudaCopyHostToDevice(context, biasDevice, bias.data(), biasBytes),
            "bias upload") &&
      check(context,
            tfCudaLinearF32(context, static_cast<const float *>(inputDevice),
                            static_cast<const float *>(weightDevice),
                            static_cast<const float *>(biasDevice),
                            static_cast<float *>(scratchDevice), m, k, n),
            "unfused Linear") &&
      check(context,
            tfCudaGeluF32(context, static_cast<const float *>(scratchDevice),
                          static_cast<float *>(unfusedDevice), m * n),
            "unfused GELU") &&
      check(context,
            tfCudaFusedLinearGeluF32(
                context, static_cast<const float *>(inputDevice),
                static_cast<const float *>(weightDevice),
                static_cast<const float *>(biasDevice),
                static_cast<float *>(fusedDevice), m, k, n),
            "fused Linear+GELU");

  std::vector<float> fused(expected.size());
  std::vector<float> unfused(expected.size());
  if (succeeded)
    succeeded =
        check(context,
              tfCudaCopyDeviceToHost(context, fused.data(), fusedDevice,
                                     outputBytes),
              "fused output download") &&
        check(context,
              tfCudaCopyDeviceToHost(context, unfused.data(), unfusedDevice,
                                     outputBytes),
              "unfused output download") &&
        check(context, tfCudaSynchronize(context), "stream synchronization");

  if (succeeded) {
    for (size_t index = 0; index < expected.size(); ++index) {
      const float tolerance =
          3.0e-4f * std::max(1.0f, std::abs(expected[index]));
      if (std::abs(fused[index] - expected[index]) > tolerance ||
          std::abs(unfused[index] - expected[index]) > tolerance) {
        std::cerr << "Linear+GELU mismatch at element " << index
                  << ": expected=" << expected[index]
                  << " fused=" << fused[index] << " unfused=" << unfused[index]
                  << '\n';
        succeeded = false;
        break;
      }
    }
  }

  float fusedMilliseconds = 0.0f;
  float unfusedMilliseconds = 0.0f;
  if (succeeded)
    succeeded =
        check(context,
              tfCudaTimeLinearGeluF32(context,
                                      static_cast<const float *>(inputDevice),
                                      static_cast<const float *>(weightDevice),
                                      static_cast<const float *>(biasDevice),
                                      static_cast<float *>(scratchDevice),
                                      static_cast<float *>(fusedDevice), m, k,
                                      n, 1, 3, &fusedMilliseconds),
              "fused timing") &&
        check(context,
              tfCudaTimeLinearGeluF32(context,
                                      static_cast<const float *>(inputDevice),
                                      static_cast<const float *>(weightDevice),
                                      static_cast<const float *>(biasDevice),
                                      static_cast<float *>(scratchDevice),
                                      static_cast<float *>(unfusedDevice), m, k,
                                      n, 0, 3, &unfusedMilliseconds),
              "unfused timing") &&
        fusedMilliseconds > 0.0f && unfusedMilliseconds > 0.0f;

  for (void *buffer : {unfusedDevice, fusedDevice, scratchDevice, biasDevice,
                       weightDevice, inputDevice}) {
    if (buffer)
      succeeded = check(context, tfCudaFree(context, buffer), "buffer free") &&
                  succeeded;
  }
  succeeded = check(context, tfCudaDestroy(context), "context destruction") &&
              succeeded;

  if (succeeded)
    std::cout << "PASS: fused and unfused Linear+GELU match; fused="
              << fusedMilliseconds << " ms, unfused=" << unfusedMilliseconds
              << " ms\n";
  return succeeded ? 0 : 1;
}
