#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainA_pred_decrease_times_two.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ChainAPredDecreaseTimesTwoKernel(
    double* ChainA_step, unsigned int ChainA_step_num_alloc, double* ChainA_precond_diag,
    unsigned int ChainA_precond_diag_num_alloc, const double* const diag, double* ChainA_njtr,
    unsigned int ChainA_njtr_num_alloc, double* const out_ChainA_pred_dec, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  __shared__ double out_ChainA_pred_dec_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(ChainA_step, 0 * ChainA_step_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(ChainA_njtr, 0 * ChainA_njtr_num_alloc,
                                            global_thread_idx, r2, r3);
    ReadIdx2<1024, double, double, double2>(ChainA_precond_diag, 0 * ChainA_precond_diag_num_alloc,
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
  };
  SumStore<double>(out_ChainA_pred_dec_local, (double*)inout_shared, 0,
                   global_thread_idx < problem_size, r2);
  SumFlushFinal<double>(out_ChainA_pred_dec_local, out_ChainA_pred_dec, 1);
}

void ChainAPredDecreaseTimesTwo(double* ChainA_step, unsigned int ChainA_step_num_alloc,
                                double* ChainA_precond_diag,
                                unsigned int ChainA_precond_diag_num_alloc,
                                const double* const diag, double* ChainA_njtr,
                                unsigned int ChainA_njtr_num_alloc,
                                double* const out_ChainA_pred_dec, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainAPredDecreaseTimesTwoKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAPredDecreaseTimesTwoKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainAPredDecreaseTimesTwoKernel<<<n_blocks, 1024, 8192>>>(
      ChainA_step, ChainA_step_num_alloc, ChainA_precond_diag, ChainA_precond_diag_num_alloc, diag,
      ChainA_njtr, ChainA_njtr_num_alloc, out_ChainA_pred_dec, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainAPredDecreaseTimesTwoKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar