#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainALinkResJac(double* const a, unsigned int a_num_alloc, double* weight,
                      unsigned int weight_num_alloc, double* offset, unsigned int offset_num_alloc,
                      double* out_res, unsigned int out_res_num_alloc, double* out_a_tridig_U,
                      unsigned int out_a_tridig_U_num_alloc, double* const out_a_njtr,
                      unsigned int out_a_njtr_num_alloc, double* const out_a_precond_diag,
                      unsigned int out_a_precond_diag_num_alloc, double* const out_a_precond_tril,
                      unsigned int out_a_precond_tril_num_alloc, size_t problem_size);

}  // namespace caspar