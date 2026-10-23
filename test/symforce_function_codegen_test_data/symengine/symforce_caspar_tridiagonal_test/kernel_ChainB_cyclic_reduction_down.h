#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBCyclicReductionDown(double* eqk, unsigned int eqk_num_alloc, int eqk_offset,
                               int eqk_stride, double* const eq, unsigned int eq_num_alloc,
                               int eq_offset, int eq_stride, double* eq_marked,
                               unsigned int eq_marked_num_alloc, int eq_marked_offset,
                               int eq_marked_stride, size_t problem_size);

}  // namespace caspar