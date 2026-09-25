#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_rosenbrock_jtjnjtr_direct.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    RosenbrockJtjnjtrDirectKernel(double* x_njtr, unsigned int x_njtr_num_alloc,
                                  SharedIndex* x_njtr_indices, double* x_jac,
                                  unsigned int x_jac_num_alloc, double* const out_x_njtr,
                                  unsigned int out_x_njtr_num_alloc, size_t problem_size) {}

void RosenbrockJtjnjtrDirect(double* x_njtr, unsigned int x_njtr_num_alloc,
                             SharedIndex* x_njtr_indices, double* x_jac,
                             unsigned int x_jac_num_alloc, double* const out_x_njtr,
                             unsigned int out_x_njtr_num_alloc, size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  RosenbrockJtjnjtrDirectKernel<<<n_blocks, 1024>>>(x_njtr, x_njtr_num_alloc, x_njtr_indices, x_jac,
                                                    x_jac_num_alloc, out_x_njtr,
                                                    out_x_njtr_num_alloc, problem_size);
}

}  // namespace caspar