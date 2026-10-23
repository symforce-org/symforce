#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_chain_coupling_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ChainCouplingJtjnjtrDirectKernel(
    double* a_njtr, unsigned int a_njtr_num_alloc, SharedIndex* a_njtr_indices, double* a_jac,
    unsigned int a_jac_num_alloc, double* b_njtr, unsigned int b_njtr_num_alloc,
    SharedIndex* b_njtr_indices, double* b_jac, unsigned int b_jac_num_alloc,
    double* const out_a_njtr, unsigned int out_a_njtr_num_alloc, double* const out_b_njtr,
    unsigned int out_b_njtr_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];
  constexpr unsigned int a_njtr_indices_shared_offset = 16384;
  constexpr unsigned int b_njtr_indices_shared_offset = 24576;

  SharedIndex* const a_njtr_indices_loc =
      reinterpret_cast<SharedIndex*>(inout_shared + a_njtr_indices_shared_offset);
  a_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? a_njtr_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  SharedIndex* const b_njtr_indices_loc =
      reinterpret_cast<SharedIndex*>(inout_shared + b_njtr_indices_shared_offset);
  b_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? b_njtr_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0;
  LoadShared<2, double, double>(b_njtr, 0 * b_njtr_num_alloc, b_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, b_njtr_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(b_jac, 2 * b_jac_num_alloc, global_thread_idx, r2, r3);
  };
  LoadShared<1, double, double>(b_njtr, 2 * b_njtr_num_alloc, b_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, b_njtr_indices_loc[threadIdx.x].target, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(b_jac, 4 * b_jac_num_alloc, global_thread_idx, r5, r6);
    r7 = fma(r4, r5, r1 * r2);
    ReadIdx2<1024, double, double, double2>(b_jac, 0 * b_jac_num_alloc, global_thread_idx, r8, r9);
    r7 = fma(r0, r8, r7);
    r4 = fma(r4, r6, r1 * r3);
    r4 = fma(r0, r9, r4);
    WriteSum2<double, double>((double*)inout_shared, r7, r4);
  };
  FlushSumShared<2, double>(out_a_njtr, 0 * out_a_njtr_num_alloc, a_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(a_njtr, 0 * a_njtr_num_alloc, a_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, a_njtr_indices_loc[threadIdx.x].target, r4, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = fma(r7, r9, r4 * r8);
    r3 = fma(r7, r3, r4 * r2);
    WriteSum2<double, double>((double*)inout_shared, r9, r3);
  };
  FlushSumShared<2, double>(out_b_njtr, 0 * out_b_njtr_num_alloc, b_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r6, r4 * r5);
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_b_njtr, 2 * out_b_njtr_num_alloc, b_njtr_indices_loc,
                            (double*)inout_shared);
}

void ChainCouplingJtjnjtrDirect(double* a_njtr, unsigned int a_njtr_num_alloc,
                                SharedIndex* a_njtr_indices, double* a_jac,
                                unsigned int a_jac_num_alloc, double* b_njtr,
                                unsigned int b_njtr_num_alloc, SharedIndex* b_njtr_indices,
                                double* b_jac, unsigned int b_jac_num_alloc,
                                double* const out_a_njtr, unsigned int out_a_njtr_num_alloc,
                                double* const out_b_njtr, unsigned int out_b_njtr_num_alloc,
                                size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainCouplingJtjnjtrDirectKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainCouplingJtjnjtrDirectKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainCouplingJtjnjtrDirectKernel<<<n_blocks, 1024, 32768>>>(
      a_njtr, a_njtr_num_alloc, a_njtr_indices, a_jac, a_jac_num_alloc, b_njtr, b_njtr_num_alloc,
      b_njtr_indices, b_jac, b_jac_num_alloc, out_a_njtr, out_a_njtr_num_alloc, out_b_njtr,
      out_b_njtr_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainCouplingJtjnjtrDirectKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar