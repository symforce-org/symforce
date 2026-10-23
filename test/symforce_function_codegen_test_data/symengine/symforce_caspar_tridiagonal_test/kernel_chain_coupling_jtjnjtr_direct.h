#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainCouplingJtjnjtrDirect(double* a_njtr, unsigned int a_njtr_num_alloc,
                                SharedIndex* a_njtr_indices, double* a_jac,
                                unsigned int a_jac_num_alloc, double* b_njtr,
                                unsigned int b_njtr_num_alloc, SharedIndex* b_njtr_indices,
                                double* b_jac, unsigned int b_jac_num_alloc,
                                double* const out_a_njtr, unsigned int out_a_njtr_num_alloc,
                                double* const out_b_njtr, unsigned int out_b_njtr_num_alloc,
                                size_t problem_size);

}  // namespace caspar