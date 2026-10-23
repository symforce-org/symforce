#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t ChainAStackedToCaspar(const double* stacked_data, double* cas_data,
                                  const unsigned int cas_stride, const unsigned int cas_offset,
                                  const unsigned int num_objects);

cudaError_t ChainACasparToStacked(const double* cas_data, double* stacked_data,
                                  const unsigned int cas_stride, const unsigned int cas_offset,
                                  const unsigned int num_objects);

cudaError_t ChainBStackedToCaspar(const double* stacked_data, double* cas_data,
                                  const unsigned int cas_stride, const unsigned int cas_offset,
                                  const unsigned int num_objects);

cudaError_t ChainBCasparToStacked(const double* cas_data, double* stacked_data,
                                  const unsigned int cas_stride, const unsigned int cas_offset,
                                  const unsigned int num_objects);

cudaError_t Matrix21StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix21CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix22StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix22CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix23StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix23CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix31StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix31CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix33StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

cudaError_t Matrix33CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects);

}  // namespace caspar