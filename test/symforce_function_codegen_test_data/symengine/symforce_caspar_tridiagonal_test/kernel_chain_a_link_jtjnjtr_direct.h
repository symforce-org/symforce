#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainALinkJtjnjtrDirect(double* const a_njtr, unsigned int a_njtr_num_alloc,
                             double* a_jac_first, unsigned int a_jac_first_num_alloc,
                             double* a_jac_second, unsigned int a_jac_second_num_alloc,
                             double* const out_a_njtr, unsigned int out_a_njtr_num_alloc,
                             size_t problem_size);

}  // namespace caspar