#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainB_pred_decrease_times_two.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ChainBPredDecreaseTimesTwoKernel(
    double* ChainB_step, unsigned int ChainB_step_num_alloc, double* ChainB_precond_diag,
    unsigned int ChainB_precond_diag_num_alloc, const double* const diag, double* ChainB_njtr,
    unsigned int ChainB_njtr_num_alloc, double* const out_ChainB_pred_dec, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  __shared__ double out_ChainB_pred_dec_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0, r9 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(ChainB_step, 0 * ChainB_step_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(ChainB_njtr, 0 * ChainB_njtr_num_alloc,
                                            global_thread_idx, r2, r3);
    ReadIdx2<1024, double, double, double2>(ChainB_precond_diag, 0 * ChainB_precond_diag_num_alloc,
                                            global_thread_idx, r4, r5);
    r6 = r0 * r4;
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r6, r2);
    r2 = r1 * r5;
    r2 = fma(r7, r2, r3);
    r2 = fma(r1, r2, r0 * r6);
    ReadIdx1<1024, double, double, double>(ChainB_step, 2 * ChainB_step_num_alloc,
                                           global_thread_idx, r6);
    ReadIdx1<1024, double, double, double>(ChainB_njtr, 2 * ChainB_njtr_num_alloc,
                                           global_thread_idx, r3);
    ReadIdx1<1024, double, double, double>(ChainB_precond_diag, 2 * ChainB_precond_diag_num_alloc,
                                           global_thread_idx, r8);
    r9 = r6 * r8;
    r9 = fma(r7, r9, r3);
    r2 = fma(r6, r9, r2);
  };
  SumStore<double>(out_ChainB_pred_dec_local, (double*)inout_shared, 0,
                   global_thread_idx < problem_size, r2);
  SumFlushFinal<double>(out_ChainB_pred_dec_local, out_ChainB_pred_dec, 1);
}

void ChainBPredDecreaseTimesTwo(double* ChainB_step, unsigned int ChainB_step_num_alloc,
                                double* ChainB_precond_diag,
                                unsigned int ChainB_precond_diag_num_alloc,
                                const double* const diag, double* ChainB_njtr,
                                unsigned int ChainB_njtr_num_alloc,
                                double* const out_ChainB_pred_dec, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBPredDecreaseTimesTwoKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBPredDecreaseTimesTwoKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBPredDecreaseTimesTwoKernel<<<n_blocks, 1024, 8192>>>(
      ChainB_step, ChainB_step_num_alloc, ChainB_precond_diag, ChainB_precond_diag_num_alloc, diag,
      ChainB_njtr, ChainB_njtr_num_alloc, out_ChainB_pred_dec, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBPredDecreaseTimesTwoKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar