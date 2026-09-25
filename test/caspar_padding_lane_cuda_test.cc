/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "kernel_caspar_padding_lane_score.h"
#include "shared_indices.h"

namespace {

// Enough for any argument's caspar layout.
constexpr size_t kSlotsPerFactor = 16;

constexpr size_t kNumNodes = 64;

float* AllocateFilled(const size_t num_factors) {
  std::vector<float> host(num_factors * kSlotsPerFactor);
  for (size_t i = 0; i < host.size(); i++) {
    host[i] = 0.3f + 0.013f * static_cast<float>(i % 19);
  }
  float* device = nullptr;
  cudaMalloc(&device, host.size() * sizeof(float));
  cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
  return device;
}

caspar::SharedIndex* AllocateIndices(const size_t problem_size) {
  std::vector<unsigned int> host(problem_size);
  for (size_t i = 0; i < problem_size; i++) {
    host[i] = static_cast<unsigned int>(i % kNumNodes);
  }
  unsigned int* device_plain = nullptr;
  cudaMalloc(&device_plain, problem_size * sizeof(unsigned int));
  cudaMemcpy(device_plain, host.data(), problem_size * sizeof(unsigned int),
             cudaMemcpyHostToDevice);

  caspar::SharedIndex* device_shared = nullptr;
  cudaMalloc(&device_shared, problem_size * sizeof(caspar::SharedIndex));
  caspar::SharedIndices(device_plain, device_shared, static_cast<unsigned int>(problem_size));
  cudaDeviceSynchronize();
  cudaFree(device_plain);
  return device_shared;
}

bool HaveDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

float Score(const size_t problem_size) {
  const size_t num_alloc = problem_size + 1024;

  // The inputs are read-only and filled identically, so one buffer serves for all of them.
  float* data = AllocateFilled(num_alloc);
  caspar::SharedIndex* indices = AllocateIndices(problem_size);

  float* out = nullptr;
  cudaMalloc(&out, sizeof(float));
  cudaMemset(out, 0, sizeof(float));

  const unsigned int num_alloc_arg = static_cast<unsigned int>(num_alloc);
  caspar::CasparPaddingLaneScore(data, num_alloc_arg, indices, data, num_alloc_arg, indices, data,
                                 num_alloc_arg, indices, data, num_alloc_arg, out, problem_size);

  float score = nanf("");
  if (cudaDeviceSynchronize() == cudaSuccess) {
    cudaMemcpy(&score, out, sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaFree(data);
  cudaFree(indices);
  cudaFree(out);
  return score;
}

}  // namespace

// Sizes that aren't multiples of the 1024-thread block leave padding lanes.  Without
// zero-initialized registers, clang drops SumStore's mask on them and the score is NaN; nvcc keeps
// it either way.
TEST_CASE("Score ignores the padding lanes of a partial block", "[caspar_padding_lane_cuda_test]") {
  REQUIRE(HaveDevice());

  for (const size_t problem_size : {1000, 1024, 1500, 2049, 5000}) {
    CAPTURE(problem_size);
    CHECK(std::isfinite(Score(problem_size)));
  }
}
