#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAUpdateP(double* ChainA_z, unsigned int ChainA_z_num_alloc, double* ChainA_p_k,
                   unsigned int ChainA_p_k_num_alloc, const double* const beta,
                   double* out_ChainA_p_kp1, unsigned int out_ChainA_p_kp1_num_alloc,
                   size_t problem_size);

}  // namespace caspar