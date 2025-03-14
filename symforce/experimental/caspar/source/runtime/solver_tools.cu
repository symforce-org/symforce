/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2025, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "solver_tools.h"

namespace caspar {

void zero(float* start, float* end) {
  const size_t num_bytes = (end - start) * sizeof(float);
  cudaMemset(start, 0, num_bytes);
}

void copy(const float* start, const float* end, float* target) {
  const size_t num_bytes = (end - start) * sizeof(float);
  cudaMemcpy(target, start, num_bytes, cudaMemcpyDeviceToDevice);
}

float sum(const float* start, const float* end, float* target_ptr, float* scratch_ptr,
          const bool copy_to_host) {
  const size_t num_el = (end - start);
  size_t tmp_storage_bytes;
  cudaMemset(target_ptr, 0, sizeof(float));
  cub::DeviceReduce::Sum(nullptr, tmp_storage_bytes, start, target_ptr, num_el);
  cub::DeviceReduce::Sum(scratch_ptr, tmp_storage_bytes, start, target_ptr, num_el);
  float result = 0.0f;
  if (copy_to_host) {
    cudaMemcpy(&result, target_ptr, sizeof(float), cudaMemcpyDeviceToHost);
  }
  return result;
}

__global__ void alpha_from_num_denum_kernel(const float* alpha_numerator,
                                            const float* alpha_denumerator, float* alpha,
                                            float* neg_alpha) {
  *alpha = *alpha_numerator / *alpha_denumerator;
  *neg_alpha = -*alpha;
}

void alpha_from_num_denum(const float* alpha_numerator, const float* alpha_denumerator,
                          float* alpha, float* neg_alpha) {
  alpha_from_num_denum_kernel<<<1, 1>>>(alpha_numerator, alpha_denumerator, alpha, neg_alpha);
}

__global__ void beta_from_num_denum_kernel(const float* beta_num, const float* beta_denum,
                                           float* beta) {
  *beta = *beta_num / *beta_denum;
}

void beta_from_num_denum(const float* beta_num, const float* beta_denum, float* beta) {
  beta_from_num_denum_kernel<<<1, 1>>>(beta_num, beta_denum, beta);
}

}  // namespace caspar
