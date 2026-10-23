#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainCouplingScore(double* a, unsigned int a_num_alloc, SharedIndex* a_indices, double* b,
                        unsigned int b_num_alloc, SharedIndex* b_indices, double* weight,
                        unsigned int weight_num_alloc, double* offset,
                        unsigned int offset_num_alloc, double* const out_rTr, size_t problem_size);

}  // namespace caspar