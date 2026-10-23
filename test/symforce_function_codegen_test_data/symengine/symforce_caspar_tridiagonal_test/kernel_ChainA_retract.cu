#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_ChainA_retract.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ChainARetractKernel(double* ChainA, unsigned int ChainA_num_alloc, double* delta,
                        unsigned int delta_num_alloc, double* out_ChainA_retracted,
                        unsigned int out_ChainA_retracted_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(ChainA, 0 * ChainA_num_alloc, global_thread_idx, r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(delta, 0 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    WriteIdx2<1024, double, double, double2>(
        out_ChainA_retracted, 0 * out_ChainA_retracted_num_alloc, global_thread_idx, r2, r3);
  };
}

void ChainARetract(double* ChainA, unsigned int ChainA_num_alloc, double* delta,
                   unsigned int delta_num_alloc, double* out_ChainA_retracted,
                   unsigned int out_ChainA_retracted_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ChainARetractKernel<<<n_blocks, 1024, 0>>>(ChainA, ChainA_num_alloc, delta, delta_num_alloc,
                                             out_ChainA_retracted, out_ChainA_retracted_num_alloc,
                                             problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("ChainARetractKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar