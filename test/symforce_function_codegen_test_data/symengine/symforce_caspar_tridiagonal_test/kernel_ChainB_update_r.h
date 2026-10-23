#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBUpdateR(double* ChainB_r_k, unsigned int ChainB_r_k_num_alloc, double* ChainB_w,
                   unsigned int ChainB_w_num_alloc, const double* const negalpha,
                   double* out_ChainB_r_kp1, unsigned int out_ChainB_r_kp1_num_alloc,
                   double* const out_ChainB_r_kp1_norm2_tot, size_t problem_size);

}  // namespace caspar