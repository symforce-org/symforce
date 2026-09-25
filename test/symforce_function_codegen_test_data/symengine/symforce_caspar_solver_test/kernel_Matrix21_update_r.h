#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21UpdateR(double* Matrix21_r_k, unsigned int Matrix21_r_k_num_alloc, double* Matrix21_w,
                     unsigned int Matrix21_w_num_alloc, const double* const negalpha,
                     double* out_Matrix21_r_kp1, unsigned int out_Matrix21_r_kp1_num_alloc,
                     double* const out_Matrix21_r_kp1_norm2_tot, size_t problem_size);

}  // namespace caspar