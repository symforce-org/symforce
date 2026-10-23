#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAPriorScore(double* a, unsigned int a_num_alloc, SharedIndex* a_indices, double* target,
                      unsigned int target_num_alloc, double* const out_rTr, size_t problem_size);

}  // namespace caspar