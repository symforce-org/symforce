#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "caspar_mappings.h"

namespace cg = cooperative_groups;

// We use shared memory to improve the memory access.
// A smaller block size of 32 allows for larger nodetypes.
constexpr int block_size = 32;

namespace caspar {

__global__ __launch_bounds__(block_size, 1) void Matrix21StackedToCaspar_kernel(
    const double* const __restrict__ stacked_data, double* const __restrict__ cas_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2; target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 2] = stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Matrix21CasparToStacked_kernel(
    const double* const __restrict__ cas_data, double* const __restrict__ stacked_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 2];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 2;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 2; target += blockDim.x) {
    stacked_data[target] = stacked_data_local[target - (blockIdx.x * blockDim.x) * 2];
  }
}

cudaError_t Matrix21StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix21StackedToCaspar_kernel<<<num_blocks, block_size>>>(stacked_data, cas_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Matrix21CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix21CasparToStacked_kernel<<<num_blocks, block_size>>>(cas_data, stacked_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

}  // namespace caspar