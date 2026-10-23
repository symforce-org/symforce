#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_b_prior_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainBPriorJtjnjtrDirectKernel(double* b_njtr, unsigned int b_njtr_num_alloc,
                                   SharedIndex* b_njtr_indices, double* b_jac,
                                   unsigned int b_jac_num_alloc, double* const out_b_njtr,
                                   unsigned int out_b_njtr_num_alloc, size_t problem_size) {}

void ChainBPriorJtjnjtrDirect(double* b_njtr, unsigned int b_njtr_num_alloc,
                              SharedIndex* b_njtr_indices, double* b_jac,
                              unsigned int b_jac_num_alloc, double* const out_b_njtr,
                              unsigned int out_b_njtr_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBPriorJtjnjtrDirectKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBPriorJtjnjtrDirectKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBPriorJtjnjtrDirectKernel<<<n_blocks, 1024, 24576>>>(
      b_njtr, b_njtr_num_alloc, b_njtr_indices, b_jac, b_jac_num_alloc, out_b_njtr,
      out_b_njtr_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBPriorJtjnjtrDirectKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar