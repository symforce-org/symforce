#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_rosenbrock_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    RosenbrockResJacKernel(double* x, unsigned int x_num_alloc, SharedIndex* x_indices,
                           double* out_res, unsigned int out_res_num_alloc,
                           double* const out_x_njtr, unsigned int out_x_njtr_num_alloc,
                           double* const out_x_precond_diag,
                           unsigned int out_x_precond_diag_num_alloc,
                           double* const out_x_precond_tril,
                           unsigned int out_x_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex x_indices_loc[1024];
  x_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size ? x_indices[global_thread_idx]
                                        : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;

  if (global_thread_idx < problem_size) {
    r0 = 1.00000000000000000e+01;
  };
  LoadShared<2, double, double>(x, 0 * x_num_alloc, x_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, x_indices_loc[threadIdx.x].target, r1, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = -1.00000000000000000e+00;
    r4 = r1 * r1;
    r4 = fma(r3, r4, r2);
    r4 = r0 * r4;
    r0 = 1.00000000000000000e+00;
    r3 = fma(r1, r3, r0);
    WriteIdx2<1024, double, double, double2>(out_res, 0 * out_res_num_alloc, global_thread_idx, r4,
                                             r3);
    r3 = 1.00000000000000000e+00;
    r4 = -1.00000000000000000e+00;
    r0 = fma(r1, r4, r3);
    r5 = 2.00000000000000000e+02;
    r6 = r1 * r5;
    r7 = r1 * r1;
    r4 = fma(r4, r7, r2);
    r0 = fma(r4, r6, r0);
    r6 = -1.00000000000000000e+02;
    r6 = r4 * r6;
    WriteSum2<double, double>((double*)inout_shared, r0, r6);
  };
  FlushSumShared<2, double>(out_x_njtr, 0 * out_x_njtr_num_alloc, x_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = 1.00000000000000000e+02;
    r0 = 4.00000000000000000e+02;
    r7 = fma(r0, r7, r3);
    WriteSum2<double, double>((double*)inout_shared, r7, r6);
  };
  FlushSumShared<2, double>(out_x_precond_diag, 0 * out_x_precond_diag_num_alloc, x_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = -2.00000000000000000e+02;
    r6 = r1 * r6;
    WriteSum1<double, double>((double*)inout_shared, r6);
  };
  FlushSumShared<1, double>(out_x_precond_tril, 0 * out_x_precond_tril_num_alloc, x_indices_loc,
                            (double*)inout_shared);
}

void RosenbrockResJac(double* x, unsigned int x_num_alloc, SharedIndex* x_indices, double* out_res,
                      unsigned int out_res_num_alloc, double* const out_x_njtr,
                      unsigned int out_x_njtr_num_alloc, double* const out_x_precond_diag,
                      unsigned int out_x_precond_diag_num_alloc, double* const out_x_precond_tril,
                      unsigned int out_x_precond_tril_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  RosenbrockResJacKernel<<<n_blocks, 1024>>>(x, x_num_alloc, x_indices, out_res, out_res_num_alloc,
                                             out_x_njtr, out_x_njtr_num_alloc, out_x_precond_diag,
                                             out_x_precond_diag_num_alloc, out_x_precond_tril,
                                             out_x_precond_tril_num_alloc, problem_size);
}

}  // namespace caspar