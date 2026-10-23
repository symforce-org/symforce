#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBPredDecreaseTimesTwo(double* ChainB_step, unsigned int ChainB_step_num_alloc,
                                double* ChainB_precond_diag,
                                unsigned int ChainB_precond_diag_num_alloc,
                                const double* const diag, double* ChainB_njtr,
                                unsigned int ChainB_njtr_num_alloc,
                                double* const out_ChainB_pred_dec, size_t problem_size);

}  // namespace caspar