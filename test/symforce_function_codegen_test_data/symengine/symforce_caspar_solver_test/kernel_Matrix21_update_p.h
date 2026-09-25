#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21UpdateP(double* Matrix21_z, unsigned int Matrix21_z_num_alloc, double* Matrix21_p_k,
                     unsigned int Matrix21_p_k_num_alloc, const double* const beta,
                     double* out_Matrix21_p_kp1, unsigned int out_Matrix21_p_kp1_num_alloc,
                     size_t problem_size);

}  // namespace caspar