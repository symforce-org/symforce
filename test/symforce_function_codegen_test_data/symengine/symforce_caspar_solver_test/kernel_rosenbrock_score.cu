#include <stdexcept>
#include <string>

#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_rosenbrock_score.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    RosenbrockScoreKernel(double* x, unsigned int x_num_alloc, SharedIndex* x_indices,
                          double* const out_rTr, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Scratch data, then the shared indices of each shared accessor; dynamic, so
  // that it may exceed the 48 KB static shared memory limit (see the launch
  // below).
  extern __shared__ __align__(16) uint8_t inout_shared[];
  constexpr unsigned int x_indices_shared_offset = 16384;

  SharedIndex* const x_indices_loc =
      reinterpret_cast<SharedIndex*>(inout_shared + x_indices_shared_offset);
  x_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? x_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0;

  if (global_thread_idx < problem_size) {
    r0 = 1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(x, 0 * x_num_alloc, x_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, x_indices_loc[threadIdx.x].target, r1, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = -1.00000000000000000e+00;
    r0 = fma(r1, r3, r0);
    r4 = 1.00000000000000000e+02;
    r1 = r1 * r1;
    r1 = fma(r3, r1, r2);
    r1 = r1 * r1;
    r1 = fma(r4, r1, r0 * r0);
  };
  SumStore<double>(out_rTr_local, (double*)inout_shared, 0, global_thread_idx < problem_size, r1);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void RosenbrockScore(double* x, unsigned int x_num_alloc, SharedIndex* x_indices,
                     double* const out_rTr, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  // Kernels using more than 48 KB of shared memory per block must opt in.
  static const cudaError_t attribute_error = cudaFuncSetAttribute(
      RosenbrockScoreKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 24576);
  if (attribute_error != cudaSuccess) {
    throw std::runtime_error(std::string("RosenbrockScoreKernel: ") +
                             cudaGetErrorString(attribute_error));
  }
  RosenbrockScoreKernel<<<n_blocks, 1024, 24576>>>(x, x_num_alloc, x_indices, out_rTr,
                                                   problem_size);
  // A launch that fails (e.g. from exceeding shared memory or register limits)
  // otherwise leaves its outputs untouched without any error.
  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    throw std::runtime_error(std::string("RosenbrockScoreKernel launch failed: ") +
                             cudaGetErrorString(launch_error));
  }
}

}  // namespace caspar