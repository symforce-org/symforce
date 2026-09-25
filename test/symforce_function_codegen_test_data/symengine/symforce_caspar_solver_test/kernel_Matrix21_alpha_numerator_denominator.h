#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21AlphaNumeratorDenominator(double* Matrix21_p_kp1,
                                       unsigned int Matrix21_p_kp1_num_alloc, double* Matrix21_r_k,
                                       unsigned int Matrix21_r_k_num_alloc, double* Matrix21_w,
                                       unsigned int Matrix21_w_num_alloc,
                                       double* const Matrix21_total_ag,
                                       double* const Matrix21_total_ac, size_t problem_size);

}  // namespace caspar