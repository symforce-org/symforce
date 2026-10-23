#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_a_prior_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainAPriorJtjnjtrDirectKernel(double* a_njtr, unsigned int a_njtr_num_alloc,
                                   SharedIndex* a_njtr_indices, double* a_jac,
                                   unsigned int a_jac_num_alloc, double* const out_a_njtr,
                                   unsigned int out_a_njtr_num_alloc, size_t problem_size) {}

void ChainAPriorJtjnjtrDirect(double* a_njtr, unsigned int a_njtr_num_alloc,
                              SharedIndex* a_njtr_indices, double* a_jac,
                              unsigned int a_jac_num_alloc, double* const out_a_njtr,
                              unsigned int out_a_njtr_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainAPriorJtjnjtrDirectKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAPriorJtjnjtrDirectKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainAPriorJtjnjtrDirectKernel<<<n_blocks, 1024, 24576>>>(
      a_njtr, a_njtr_num_alloc, a_njtr_indices, a_jac, a_jac_num_alloc, out_a_njtr,
      out_a_njtr_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAPriorJtjnjtrDirectKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar