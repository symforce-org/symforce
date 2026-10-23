#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBPriorResJac(double* b, unsigned int b_num_alloc, SharedIndex* b_indices, double* target,
                       unsigned int target_num_alloc, double* out_res,
                       unsigned int out_res_num_alloc, double* const out_b_njtr,
                       unsigned int out_b_njtr_num_alloc, double* const out_b_precond_diag,
                       unsigned int out_b_precond_diag_num_alloc, double* const out_b_precond_tril,
                       unsigned int out_b_precond_tril_num_alloc, size_t problem_size);

}  // namespace caspar