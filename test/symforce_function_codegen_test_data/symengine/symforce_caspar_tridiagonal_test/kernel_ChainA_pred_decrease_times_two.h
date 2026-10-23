#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAPredDecreaseTimesTwo(double* ChainA_step, unsigned int ChainA_step_num_alloc,
                                double* ChainA_precond_diag,
                                unsigned int ChainA_precond_diag_num_alloc,
                                const double* const diag, double* ChainA_njtr,
                                unsigned int ChainA_njtr_num_alloc,
                                double* const out_ChainA_pred_dec, size_t problem_size);

}  // namespace caspar