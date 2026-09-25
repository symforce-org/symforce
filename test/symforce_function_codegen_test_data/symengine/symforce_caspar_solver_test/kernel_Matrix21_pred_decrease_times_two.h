#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void Matrix21PredDecreaseTimesTwo(double* Matrix21_step, unsigned int Matrix21_step_num_alloc,
                                  double* Matrix21_precond_diag,
                                  unsigned int Matrix21_precond_diag_num_alloc,
                                  const double* const diag, double* Matrix21_njtr,
                                  unsigned int Matrix21_njtr_num_alloc,
                                  double* const out_Matrix21_pred_dec, size_t problem_size);

}  // namespace caspar