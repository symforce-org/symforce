#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21StartW(double* Matrix21_precond_diag, unsigned int Matrix21_precond_diag_num_alloc,
                    const double* const diag, double* Matrix21_p, unsigned int Matrix21_p_num_alloc,
                    double* out_Matrix21_w, unsigned int out_Matrix21_w_num_alloc,
                    size_t problem_size);

}  // namespace caspar