#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainACyclicReductionSolveTwo(double* const eq, unsigned int eq_num_alloc, int eq_offset,
                                   int eq_stride, double* solu2km1, unsigned int solu2km1_num_alloc,
                                   int solu2km1_offset, int solu2km1_stride, double* solu2k,
                                   unsigned int solu2k_num_alloc, int solu2k_offset,
                                   int solu2k_stride, size_t problem_size);

}  // namespace caspar