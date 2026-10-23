#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainARetract(double* ChainA, unsigned int ChainA_num_alloc, double* delta,
                   unsigned int delta_num_alloc, double* out_ChainA_retracted,
                   unsigned int out_ChainA_retracted_num_alloc, size_t problem_size);

}  // namespace caspar