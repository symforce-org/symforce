#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAAlphaDenominatorOrBetaNumerator(double* ChainA_p_kp1,
                                           unsigned int ChainA_p_kp1_num_alloc, double* ChainA_w,
                                           unsigned int ChainA_w_num_alloc,
                                           double* const ChainA_out, size_t problem_size);

}  // namespace caspar