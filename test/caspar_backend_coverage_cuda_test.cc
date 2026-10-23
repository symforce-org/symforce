/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <algorithm>
#include <cmath>
#include <iterator>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "backend_test_function_expected.h"
#include "kernel_backend_test_function.h"

namespace {

bool HaveDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

double* CopyToDevice(const std::vector<double>& host) {
  double* device = nullptr;
  cudaMalloc(&device, host.size() * sizeof(double));
  cudaMemcpy(device, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice);
  return device;
}

}  // namespace

TEST_CASE("The Caspar backend test function matches SymPy at every point",
          "[caspar_backend_coverage_cuda_test]") {
  REQUIRE(HaveDevice());

  using caspar_backend_coverage::kExpected;
  using caspar_backend_coverage::kNumOutputs;
  using caspar_backend_coverage::kNumPoints;

  double* const x = CopyToDevice(
      {std::begin(caspar_backend_coverage::kX), std::end(caspar_backend_coverage::kX)});
  double* const y = CopyToDevice(
      {std::begin(caspar_backend_coverage::kY), std::end(caspar_backend_coverage::kY)});
  // The caspar layout of doubles: chunks of two outputs, each holding both for every point.
  static_assert(kNumOutputs % 2 == 0);
  std::vector<double> out_host(kNumOutputs * kNumPoints, NAN);
  double* const out = CopyToDevice(out_host);

  caspar::BackendTestFunction(x, kNumPoints, y, kNumPoints, out, kNumPoints, kNumPoints);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  cudaMemcpy(out_host.data(), out, out_host.size() * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(x);
  cudaFree(y);
  cudaFree(out);

  for (int output = 0; output < kNumOutputs; output++) {
    for (int point = 0; point < kNumPoints; point++) {
      const int chunk_start = output - output % 2;
      const double actual = out_host.at(chunk_start * kNumPoints + point * 2 + output % 2);
      const double expected = kExpected[output][point];
      CAPTURE(output, point, caspar_backend_coverage::kX[point], caspar_backend_coverage::kY[point],
              expected, actual);
      if (std::isnan(expected)) {
        CHECK(std::isnan(actual));
      } else {
        CHECK(std::abs(actual - expected) <= 1e-12 * std::max(1.0, std::abs(expected)));
      }
    }
  }
}
