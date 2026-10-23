#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAStartWContribute(double* ChainA_precond_diag, unsigned int ChainA_precond_diag_num_alloc,
                            const double* const diag, double* ChainA_p,
                            unsigned int ChainA_p_num_alloc, double* out_ChainA_w,
                            unsigned int out_ChainA_w_num_alloc, size_t problem_size);

}  // namespace caspar