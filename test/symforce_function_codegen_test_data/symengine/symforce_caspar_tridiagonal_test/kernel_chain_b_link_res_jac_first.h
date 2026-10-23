#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBLinkResJacFirst(
    double* const b, unsigned int b_num_alloc, double* weight, unsigned int weight_num_alloc,
    double* offset, unsigned int offset_num_alloc, double* out_res, unsigned int out_res_num_alloc,
    double* const out_rTr, double* out_b_tridig_U, unsigned int out_b_tridig_U_num_alloc,
    double* const out_b_njtr, unsigned int out_b_njtr_num_alloc, double* const out_b_precond_diag,
    unsigned int out_b_precond_diag_num_alloc, double* const out_b_precond_tril,
    unsigned int out_b_precond_tril_num_alloc, size_t problem_size);

}  // namespace caspar