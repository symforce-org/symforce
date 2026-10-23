#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_coupling_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainCouplingScoreKernel(double* a, unsigned int a_num_alloc, SharedIndex* a_indices, double* b,
                             unsigned int b_num_alloc, SharedIndex* b_indices, double* weight,
                             unsigned int weight_num_alloc, double* offset,
                             unsigned int offset_num_alloc, double* const out_rTr,
                             size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];
  constexpr unsigned int a_indices_shared_offset = 16384;
  constexpr unsigned int b_indices_shared_offset = 24576;

  SharedIndex* const a_indices_loc =
      reinterpret_cast<SharedIndex*>(inout_shared + a_indices_shared_offset);
  a_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? a_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});
  SharedIndex* const b_indices_loc =
      reinterpret_cast<SharedIndex*>(inout_shared + b_indices_shared_offset);
  b_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? b_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0, r10 = 0,
         r11 = 0;
  LoadShared<2, double, double>(a, 0 * a_num_alloc, a_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, a_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(offset, 0 * offset_num_alloc, global_thread_idx, r2,
                                            r3);
    r4 = -1.00000000000000000e+00;
    r3 = fma(r3, r4, r1);
  };
  LoadShared<1, double, double>(b, 2 * b_num_alloc, b_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, b_indices_loc[threadIdx.x].target, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(weight, 4 * weight_num_alloc, global_thread_idx, r5,
                                            r6);
  };
  LoadShared<2, double, double>(b, 0 * b_num_alloc, b_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, b_indices_loc[threadIdx.x].target, r7, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(weight, 2 * weight_num_alloc, global_thread_idx, r9,
                                            r10);
    r10 = fma(r8, r10, r1 * r6);
    ReadIdx2<1024, double, double, double2>(weight, 0 * weight_num_alloc, global_thread_idx, r6,
                                            r11);
    r10 = fma(r7, r11, r10);
    r3 = fma(r4, r10, r3);
    r2 = fma(r2, r4, r0);
    r9 = fma(r8, r9, r1 * r5);
    r9 = fma(r7, r6, r9);
    r2 = fma(r4, r9, r2);
    r2 = fma(r2, r2, r3 * r3);
  };
  SumStore<double>(out_rTr_local, (double*)inout_shared, 0, global_thread_idx < problem_size, r2);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ChainCouplingScore(double* a, unsigned int a_num_alloc, SharedIndex* a_indices, double* b,
                        unsigned int b_num_alloc, SharedIndex* b_indices, double* weight,
                        unsigned int weight_num_alloc, double* offset,
                        unsigned int offset_num_alloc, double* const out_rTr, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainCouplingScoreKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainCouplingScoreKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainCouplingScoreKernel<<<n_blocks, 1024, 32768>>>(a, a_num_alloc, a_indices, b, b_num_alloc,
                                                      b_indices, weight, weight_num_alloc, offset,
                                                      offset_num_alloc, out_rTr, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainCouplingScoreKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar