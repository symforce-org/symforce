#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAUpdateMp(double* ChainA_r_k, unsigned int ChainA_r_k_num_alloc, double* ChainA_Mp,
                    unsigned int ChainA_Mp_num_alloc, const double* const beta,
                    double* out_ChainA_Mp_kp1, unsigned int out_ChainA_Mp_kp1_num_alloc,
                    double* out_ChainA_w, unsigned int out_ChainA_w_num_alloc, size_t problem_size);

}  // namespace caspar