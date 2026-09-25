#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21Retract(double* Matrix21, unsigned int Matrix21_num_alloc, double* delta,
                     unsigned int delta_num_alloc, double* out_Matrix21_retracted,
                     unsigned int out_Matrix21_retracted_num_alloc, size_t problem_size);

}  // namespace caspar