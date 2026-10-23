#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBRetract(double* ChainB, unsigned int ChainB_num_alloc, double* delta,
                   unsigned int delta_num_alloc, double* out_ChainB_retracted,
                   unsigned int out_ChainB_retracted_num_alloc, size_t problem_size);

}  // namespace caspar