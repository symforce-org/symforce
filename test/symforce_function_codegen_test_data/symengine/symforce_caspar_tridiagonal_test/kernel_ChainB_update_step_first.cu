#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainB_update_step_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainBUpdateStepFirstKernel(double* ChainB_p_kp1, unsigned int ChainB_p_kp1_num_alloc,
                                const double* const alpha, double* out_ChainB_step_kp1,
                                unsigned int out_ChainB_step_kp1_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];

  double r0 = 0, r1 = 0, r2 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(ChainB_p_kp1, 0 * ChainB_p_kp1_num_alloc,
                                            global_thread_idx, r0, r1);
  };
  LoadUnique<1, double, double>(alpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r2;
    r1 = r1 * r2;
    WriteIdx2<1024, double, double, double2>(out_ChainB_step_kp1, 0 * out_ChainB_step_kp1_num_alloc,
                                             global_thread_idx, r0, r1);
    ReadIdx1<1024, double, double, double>(ChainB_p_kp1, 2 * ChainB_p_kp1_num_alloc,
                                           global_thread_idx, r1);
    r2 = r1 * r2;
    WriteIdx1<1024, double, double, double>(out_ChainB_step_kp1, 2 * out_ChainB_step_kp1_num_alloc,
                                            global_thread_idx, r2);
  };
}

void ChainBUpdateStepFirst(double* ChainB_p_kp1, unsigned int ChainB_p_kp1_num_alloc,
                           const double* const alpha, double* out_ChainB_step_kp1,
                           unsigned int out_ChainB_step_kp1_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      ChainBUpdateStepFirstKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 8192);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBUpdateStepFirstKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  ChainBUpdateStepFirstKernel<<<n_blocks, 1024, 8192>>>(
      ChainB_p_kp1, ChainB_p_kp1_num_alloc, alpha, out_ChainB_step_kp1,
      out_ChainB_step_kp1_num_alloc, problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainBUpdateStepFirstKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar