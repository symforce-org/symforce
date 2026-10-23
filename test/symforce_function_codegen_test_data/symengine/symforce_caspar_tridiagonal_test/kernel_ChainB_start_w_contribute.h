#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBStartWContribute(double* ChainB_precond_diag, unsigned int ChainB_precond_diag_num_alloc,
                            const double* const diag, double* ChainB_p,
                            unsigned int ChainB_p_num_alloc, double* out_ChainB_w,
                            unsigned int out_ChainB_w_num_alloc, size_t problem_size);

}  // namespace caspar