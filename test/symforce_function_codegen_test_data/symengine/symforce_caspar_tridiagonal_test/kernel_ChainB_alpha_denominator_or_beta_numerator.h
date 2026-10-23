#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBAlphaDenominatorOrBetaNumerator(double* ChainB_p_kp1,
                                           unsigned int ChainB_p_kp1_num_alloc, double* ChainB_w,
                                           unsigned int ChainB_w_num_alloc,
                                           double* const ChainB_out, size_t problem_size);

}  // namespace caspar