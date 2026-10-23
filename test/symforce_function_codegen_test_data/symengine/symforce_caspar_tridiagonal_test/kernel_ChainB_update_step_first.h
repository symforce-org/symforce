#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBUpdateStepFirst(double* ChainB_p_kp1, unsigned int ChainB_p_kp1_num_alloc,
                           const double* const alpha, double* out_ChainB_step_kp1,
                           unsigned int out_ChainB_step_kp1_num_alloc, size_t problem_size);

}  // namespace caspar