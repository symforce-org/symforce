#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_Matrix21_pred_decrease_times_two.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    Matrix21PredDecreaseTimesTwoKernel(double* Matrix21_step, unsigned int Matrix21_step_num_alloc,
                                       double* Matrix21_precond_diag,
                                       unsigned int Matrix21_precond_diag_num_alloc,
                                       const double* const diag, double* Matrix21_njtr,
                                       unsigned int Matrix21_njtr_num_alloc,
                                       double* const out_Matrix21_pred_dec, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_Matrix21_pred_dec_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(Matrix21_step, 0 * Matrix21_step_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(Matrix21_njtr, 0 * Matrix21_njtr_num_alloc,
                                            global_thread_idx, r2, r3);
    ReadIdx2<1024, double, double, double2>(
        Matrix21_precond_diag, 0 * Matrix21_precond_diag_num_alloc, global_thread_idx, r4, r5);
    r6 = r1 * r5;
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r6, r3);
    r3 = r0 * r4;
    r3 = fma(r7, r3, r2);
    r3 = fma(r0, r3, r1 * r6);
  };
  SumStore<double>(out_Matrix21_pred_dec_local, (double*)inout_shared, 0,
                   global_thread_idx < problem_size, r3);
  SumFlushFinal<double>(out_Matrix21_pred_dec_local, out_Matrix21_pred_dec, 1);
}

void Matrix21PredDecreaseTimesTwo(double* Matrix21_step, unsigned int Matrix21_step_num_alloc,
                                  double* Matrix21_precond_diag,
                                  unsigned int Matrix21_precond_diag_num_alloc,
                                  const double* const diag, double* Matrix21_njtr,
                                  unsigned int Matrix21_njtr_num_alloc,
                                  double* const out_Matrix21_pred_dec, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Matrix21PredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      Matrix21_step, Matrix21_step_num_alloc, Matrix21_precond_diag,
      Matrix21_precond_diag_num_alloc, diag, Matrix21_njtr, Matrix21_njtr_num_alloc,
      out_Matrix21_pred_dec, problem_size);
}

}  // namespace caspar