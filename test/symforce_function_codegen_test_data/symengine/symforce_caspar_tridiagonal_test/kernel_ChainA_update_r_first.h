#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAUpdateRFirst(double* ChainA_r_k, unsigned int ChainA_r_k_num_alloc, double* ChainA_w,
                        unsigned int ChainA_w_num_alloc, const double* const negalpha,
                        double* out_ChainA_r_kp1, unsigned int out_ChainA_r_kp1_num_alloc,
                        double* const out_ChainA_r_0_norm2_tot,
                        double* const out_ChainA_r_kp1_norm2_tot, size_t problem_size);

}  // namespace caspar