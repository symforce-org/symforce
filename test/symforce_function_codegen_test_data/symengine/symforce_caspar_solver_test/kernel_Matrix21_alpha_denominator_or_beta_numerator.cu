#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_Matrix21_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) Matrix21AlphaDenominatorOrBetaNumeratorKernel(
    double* Matrix21_p_kp1, unsigned int Matrix21_p_kp1_num_alloc, double* Matrix21_w,
    unsigned int Matrix21_w_num_alloc, double* const Matrix21_out, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double Matrix21_out_local[1];

  double r0 = 0, r1 = 0, r2 = 0, r3 = 0;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(Matrix21_p_kp1, 0 * Matrix21_p_kp1_num_alloc,
                                            global_thread_idx, r0, r1);
    ReadIdx2<1024, double, double, double2>(Matrix21_w, 0 * Matrix21_w_num_alloc, global_thread_idx,
                                            r2, r3);
    r3 = fma(r1, r3, r0 * r2);
  };
  SumStore<double>(Matrix21_out_local, (double*)inout_shared, 0, global_thread_idx < problem_size,
                   r3);
  SumFlushFinal<double>(Matrix21_out_local, Matrix21_out, 1);
}

void Matrix21AlphaDenominatorOrBetaNumerator(double* Matrix21_p_kp1,
                                             unsigned int Matrix21_p_kp1_num_alloc,
                                             double* Matrix21_w, unsigned int Matrix21_w_num_alloc,
                                             double* const Matrix21_out, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  Matrix21AlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      Matrix21_p_kp1, Matrix21_p_kp1_num_alloc, Matrix21_w, Matrix21_w_num_alloc, Matrix21_out,
      problem_size);
}

}  // namespace caspar