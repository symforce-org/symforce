#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21UpdateStep(double* Matrix21_step_k, unsigned int Matrix21_step_k_num_alloc,
                        double* Matrix21_p_kp1, unsigned int Matrix21_p_kp1_num_alloc,
                        const double* const alpha, double* out_Matrix21_step_kp1,
                        unsigned int out_Matrix21_step_kp1_num_alloc, size_t problem_size);

}  // namespace caspar