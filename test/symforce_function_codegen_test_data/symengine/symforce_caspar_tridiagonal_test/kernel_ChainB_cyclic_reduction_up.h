#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBCyclicReductionUp(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
                             int eqk_stride, double* const x, unsigned int x_num_alloc,
                             int x_offset, int x_stride, double* soluk,
                             unsigned int soluk_num_alloc, int soluk_offset, int soluk_stride,
                             size_t problem_size);

}  // namespace caspar