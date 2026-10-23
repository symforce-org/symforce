#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBLinkScore(double* const b, unsigned int b_num_alloc, double* weight,
                     unsigned int weight_num_alloc, double* offset, unsigned int offset_num_alloc,
                     double* const out_rTr, size_t problem_size);

}  // namespace caspar