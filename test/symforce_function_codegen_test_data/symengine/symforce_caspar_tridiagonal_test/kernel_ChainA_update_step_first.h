#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAUpdateStepFirst(double* ChainA_p_kp1, unsigned int ChainA_p_kp1_num_alloc,
                           const double* const alpha, double* out_ChainA_step_kp1,
                           unsigned int out_ChainA_step_kp1_num_alloc, size_t problem_size);

}  // namespace caspar