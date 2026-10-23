#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBUpdateP(double* ChainB_z, unsigned int ChainB_z_num_alloc, double* ChainB_p_k,
                   unsigned int ChainB_p_k_num_alloc, const double* const beta,
                   double* out_ChainB_p_kp1, unsigned int out_ChainB_p_kp1_num_alloc,
                   size_t problem_size);

}  // namespace caspar