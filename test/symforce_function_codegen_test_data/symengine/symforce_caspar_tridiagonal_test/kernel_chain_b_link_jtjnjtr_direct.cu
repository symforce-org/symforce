#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_b_link_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainBLinkJtjnjtrDirectKernel(double* const b_njtr, unsigned int b_njtr_num_alloc,
                                  double* b_jac_first, unsigned int b_jac_first_num_alloc,
                                  double* b_jac_second, unsigned int b_jac_second_num_alloc,
                                  double* const out_b_njtr, unsigned int out_b_njtr_num_alloc,
                                  size_t problem_size) {}

void ChainBLinkJtjnjtrDirect(double* const b_njtr, unsigned int b_njtr_num_alloc,
                             double* b_jac_first, unsigned int b_jac_first_num_alloc,
                             double* b_jac_second, unsigned int b_jac_second_num_alloc,
                             double* const out_b_njtr, unsigned int out_b_njtr_num_alloc,
                             size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBLinkJtjnjtrDirectKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16400);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBLinkJtjnjtrDirectKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBLinkJtjnjtrDirectKernel<<<n_blocks, 1024, 16400>>>(
      b_njtr, b_njtr_num_alloc, b_jac_first, b_jac_first_num_alloc, b_jac_second,
      b_jac_second_num_alloc, out_b_njtr, out_b_njtr_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBLinkJtjnjtrDirectKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar