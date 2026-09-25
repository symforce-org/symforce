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
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex x_indices_loc[1024];
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
  RosenbrockScoreKernel<<<n_blocks, 1024>>>(x, x_num_alloc, x_indices, out_rTr, problem_size);
}

}  // namespace caspar