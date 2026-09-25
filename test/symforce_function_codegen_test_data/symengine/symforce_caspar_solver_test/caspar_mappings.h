#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t Matrix21StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix21CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

}  // namespace caspar