#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAPriorResJac(double* a, unsigned int a_num_alloc, SharedIndex* a_indices, double* target,
                       unsigned int target_num_alloc, double* out_res,
                       unsigned int out_res_num_alloc, double* const out_a_njtr,
                       unsigned int out_a_njtr_num_alloc, double* const out_a_precond_diag,
                       unsigned int out_a_precond_diag_num_alloc, double* const out_a_precond_tril,
                       unsigned int out_a_precond_tril_num_alloc, size_t problem_size);

}  // namespace caspar