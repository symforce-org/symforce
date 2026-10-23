#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAAlphaNumeratorDenominator(double* ChainA_p_kp1, unsigned int ChainA_p_kp1_num_alloc,
                                     double* ChainA_r_k, unsigned int ChainA_r_k_num_alloc,
                                     double* ChainA_w, unsigned int ChainA_w_num_alloc,
                                     double* const ChainA_total_ag, double* const ChainA_total_ac,
                                     size_t problem_size);

}  // namespace caspar