#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBUpdateMp(double* ChainB_r_k, unsigned int ChainB_r_k_num_alloc, double* ChainB_Mp,
                    unsigned int ChainB_Mp_num_alloc, const double* const beta,
                    double* out_ChainB_Mp_kp1, unsigned int out_ChainB_Mp_kp1_num_alloc,
                    double* out_ChainB_w, unsigned int out_ChainB_w_num_alloc, size_t problem_size);

}  // namespace caspar