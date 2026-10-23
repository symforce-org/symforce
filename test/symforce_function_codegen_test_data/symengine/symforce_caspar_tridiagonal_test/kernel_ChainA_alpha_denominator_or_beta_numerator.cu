#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainA_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainAAlphaDenominatorOrBetaNumeratorKernel(double* ChainA_p_kp1,
                                                unsigned int ChainA_p_kp1_num_alloc,
                                                double* ChainA_w, unsigned int ChainA_w_num_alloc,
                                                double* const ChainA_out, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  __shared__ double ChainA_out_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(ChainA_p_kp1, 0 * ChainA_p_kp1_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(ChainA_w, 0 * ChainA_w_num_alloc, global_thread_idx, r2,
                                            r3);
    r2 = fma(r0, r2, r1 * r3);
  };
  SumStore<double>(ChainA_out_local, (double*)inout_shared, 0, global_thread_idx < problem_size,
                   r2);
  SumFlushFinal<double>(ChainA_out_local, ChainA_out, 1);
}

void ChainAAlphaDenominatorOrBetaNumerator(double* ChainA_p_kp1,
                                           unsigned int ChainA_p_kp1_num_alloc, double* ChainA_w,
                                           unsigned int ChainA_w_num_alloc,
                                           double* const ChainA_out, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error =
      cudaFuncSetAttribute(ChainAAlphaDenominatorOrBetaNumeratorKernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 256);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAAlphaDenominatorOrBetaNumeratorKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainAAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024, 256>>>(
      ChainA_p_kp1, ChainA_p_kp1_num_alloc, ChainA_w, ChainA_w_num_alloc, ChainA_out, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(
        std::string("ChainAAlphaDenominatorOrBetaNumeratorKernel launch failed: ") +
        cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar