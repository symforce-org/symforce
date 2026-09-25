#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21UpdateMp(double* Matrix21_r_k, unsigned int Matrix21_r_k_num_alloc,
                      double* Matrix21_Mp, unsigned int Matrix21_Mp_num_alloc,
                      const double* const beta, double* out_Matrix21_Mp_kp1,
                      unsigned int out_Matrix21_Mp_kp1_num_alloc, double* out_Matrix21_w,
                      unsigned int out_Matrix21_w_num_alloc, size_t problem_size);

}  // namespace caspar