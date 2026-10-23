#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ChainBLinkJtjnjtrDirect(double* const b_njtr, unsigned int b_njtr_num_alloc,
                             double* b_jac_first, unsigned int b_jac_first_num_alloc,
                             double* b_jac_second, unsigned int b_jac_second_num_alloc,
                             double* const out_b_njtr, unsigned int out_b_njtr_num_alloc,
                             size_t problem_size);

}  // namespace caspar