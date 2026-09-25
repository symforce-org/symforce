#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void RosenbrockJtjnjtrDirect(double* x_njtr, unsigned int x_njtr_num_alloc,
                             SharedIndex* x_njtr_indices, double* x_jac,
                             unsigned int x_jac_num_alloc, double* const out_x_njtr,
                             unsigned int out_x_njtr_num_alloc, size_t problem_size);

}  // namespace caspar