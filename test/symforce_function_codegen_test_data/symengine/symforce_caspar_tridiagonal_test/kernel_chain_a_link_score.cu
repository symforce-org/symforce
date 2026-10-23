#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_a_link_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainALinkScoreKernel(double* const a, unsigned int a_num_alloc, double* weight,
                          unsigned int weight_num_alloc, double* offset,
                          unsigned int offset_num_alloc, double* const out_rTr,
                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(offset, 0 * offset_num_alloc, global_thread_idx, r0,
                                            r1);
    r2 = -1.00000000000000000e+00;
    ReadIdx2<1024, double, double, double2>(weight, 0 * weight_num_alloc, global_thread_idx, r3,
                                            r4);
  };
  ReadAndShuffle2<1024, double, double, double2>((double*)inout_shared, a, 0 * a_num_alloc,
                                                 global_thread_idx, threadIdx.x == 1023,
                                                 global_thread_idx <= problem_size, r5, r6, r7, r8);
  if (global_thread_idx < problem_size) {
    r5 = fma(r5, r2, r7);
    r4 = fma(r4, r5, r1 * r2);
    ReadIdx2<1024, double, double, double2>(weight, 2 * weight_num_alloc, global_thread_idx, r1,
                                            r7);
    r6 = fma(r6, r2, r8);
    r4 = fma(r7, r6, r4);
    r5 = fma(r3, r5, r0 * r2);
    r5 = fma(r1, r6, r5);
    r5 = fma(r5, r5, r4 * r4);
  };
  SumStore<double>(out_rTr_local, (double*)inout_shared, 0, global_thread_idx < problem_size, r5);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ChainALinkScore(double* const a, unsigned int a_num_alloc, double* weight,
                     unsigned int weight_num_alloc, double* offset, unsigned int offset_num_alloc,
                     double* const out_rTr, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainALinkScoreKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainALinkScoreKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainALinkScoreKernel<<<n_blocks, 1024, 16384>>>(a, a_num_alloc, weight, weight_num_alloc, offset,
                                                   offset_num_alloc, out_rTr, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainALinkScoreKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar