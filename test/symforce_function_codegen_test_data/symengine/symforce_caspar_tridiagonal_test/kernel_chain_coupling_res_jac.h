#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainCouplingResJac(
    double* a, unsigned int a_num_alloc, SharedIndex* a_indices, double* b,
    unsigned int b_num_alloc, SharedIndex* b_indices, double* weight, unsigned int weight_num_alloc,
    double* offset, unsigned int offset_num_alloc, double* out_res, unsigned int out_res_num_alloc,
    double* out_a_jac, unsigned int out_a_jac_num_alloc, double* const out_a_njtr,
    unsigned int out_a_njtr_num_alloc, double* const out_a_precond_diag,
    unsigned int out_a_precond_diag_num_alloc, double* const out_a_precond_tril,
    unsigned int out_a_precond_tril_num_alloc, double* out_b_jac, unsigned int out_b_jac_num_alloc,
    double* const out_b_njtr, unsigned int out_b_njtr_num_alloc, double* const out_b_precond_diag,
    unsigned int out_b_precond_diag_num_alloc, double* const out_b_precond_tril,
    unsigned int out_b_precond_tril_num_alloc, size_t problem_size);

}  // namespace caspar