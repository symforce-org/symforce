#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainAPriorJtjnjtrDirect(double* a_njtr, unsigned int a_njtr_num_alloc,
                              SharedIndex* a_njtr_indices, double* a_jac,
                              unsigned int a_jac_num_alloc, double* const out_a_njtr,
                              unsigned int out_a_njtr_num_alloc, size_t problem_size);

}  // namespace caspar