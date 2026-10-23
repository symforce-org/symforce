#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void BackendTestFunction(double* x, unsigned int x_num_alloc, double* y, unsigned int y_num_alloc,
                         double* out, unsigned int out_num_alloc, size_t problem_size);

}  // namespace caspar