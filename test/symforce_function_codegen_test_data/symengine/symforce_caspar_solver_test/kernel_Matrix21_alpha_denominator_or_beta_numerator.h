#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21AlphaDenominatorOrBetaNumerator(double* Matrix21_p_kp1,
                                             unsigned int Matrix21_p_kp1_num_alloc,
                                             double* Matrix21_w, unsigned int Matrix21_w_num_alloc,
                                             double* const Matrix21_out, size_t problem_size);

}  // namespace caspar