#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_b_link_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainBLinkScoreKernel(double* const b, unsigned int b_num_alloc, double* weight,
                          unsigned int weight_num_alloc, double* offset,
                          unsigned int offset_num_alloc, double* const out_rTr,
                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(offset, 2 * offset_num_alloc, global_thread_idx, r0);
    r1 = -1.00000000000000000e+00;
    ReadIdx1<1024, double, double, double>(weight, 8 * weight_num_alloc, global_thread_idx, r2);
  };
  ReadAndShuffle1<1024, double, double, double>((double*)inout_shared, b, 2 * b_num_alloc,
                                                global_thread_idx, threadIdx.x == 1023,
                                                global_thread_idx <= problem_size, r3, r4);
  if (global_thread_idx < problem_size) {
    r3 = fma(r3, r1, r4);
    r2 = fma(r2, r3, r0 * r1);
    ReadIdx2<1024, double, double, double2>(weight, 2 * weight_num_alloc, global_thread_idx, r0,
                                            r4);
  };
  ReadAndShuffle2<1024, double, double, double2>((double*)inout_shared, b, 0 * b_num_alloc,
                                                 global_thread_idx, threadIdx.x == 1023,
                                                 global_thread_idx <= problem_size, r5, r6, r7, r8);
  if (global_thread_idx < problem_size) {
    r5 = fma(r5, r1, r7);
    ReadIdx2<1024, double, double, double2>(weight, 4 * weight_num_alloc, global_thread_idx, r7,
                                            r9);
    r6 = fma(r6, r1, r8);
    r2 = fma(r0, r5, r2);
    r2 = fma(r9, r6, r2);
    ReadIdx2<1024, double, double, double2>(offset, 0 * offset_num_alloc, global_thread_idx, r9,
                                            r0);
    ReadIdx2<1024, double, double, double2>(weight, 6 * weight_num_alloc, global_thread_idx, r8,
                                            r10);
    r8 = fma(r8, r3, r9 * r1);
    ReadIdx2<1024, double, double, double2>(weight, 0 * weight_num_alloc, global_thread_idx, r9,
                                            r11);
    r8 = fma(r9, r5, r8);
    r8 = fma(r4, r6, r8);
    r8 = fma(r8, r8, r2 * r2);
    r3 = fma(r10, r3, r0 * r1);
    r3 = fma(r11, r5, r3);
    r3 = fma(r7, r6, r3);
    r8 = fma(r3, r3, r8);
  };
  SumStore<double>(out_rTr_local, (double*)inout_shared, 0, global_thread_idx < problem_size, r8);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ChainBLinkScore(double* const b, unsigned int b_num_alloc, double* weight,
                     unsigned int weight_num_alloc, double* offset, unsigned int offset_num_alloc,
                     double* const out_rTr, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBLinkScoreKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBLinkScoreKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBLinkScoreKernel<<<n_blocks, 1024, 16384>>>(b, b_num_alloc, weight, weight_num_alloc, offset,
                                                   offset_num_alloc, out_rTr, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBLinkScoreKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar