#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBTridigMakeEq(double* precond_diag, unsigned int precond_diag_num_alloc,
                        double* precond_tril, unsigned int precond_tril_num_alloc,
                        double* const tridig_U, unsigned int tridig_U_num_alloc,
                        int tridig_U_offset, int tridig_U_stride, double* njtr,
                        unsigned int njtr_num_alloc, const double* const diag, double* eq,
                        unsigned int eq_num_alloc, size_t problem_size);

}  // namespace caspar