#include <stdio.h>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "caspar_mappings.h"

namespace cg = cooperative_groups;

// We use shared memory to improve the memory access.
// A smaller block size of 32 allows for larger nodetypes.
constexpr int block_size = 32;

namespace caspar {

__global__ __launch_bounds__(block_size, 1) void ChainAStackedToCaspar_kernel(
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

__global__ __launch_bounds__(block_size, 1) void ChainACasparToStacked_kernel(
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

cudaError_t ChainAStackedToCaspar(const double* stacked_data, double* cas_data,
                                  const unsigned int cas_stride, const unsigned int cas_offset,
                                  const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ChainAStackedToCaspar_kernel<<<num_blocks, block_size>>>(stacked_data, cas_data, cas_stride,
                                                           cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ChainACasparToStacked(const double* cas_data, double* stacked_data,
                                  const unsigned int cas_stride, const unsigned int cas_offset,
                                  const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ChainACasparToStacked_kernel<<<num_blocks, block_size>>>(cas_data, stacked_data, cas_stride,
                                                           cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void ChainBStackedToCaspar_kernel(
    const double* const __restrict__ stacked_data, double* const __restrict__ cas_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3; target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] = stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void ChainBCasparToStacked_kernel(
    const double* const __restrict__ cas_data, double* const __restrict__ stacked_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[2] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3; target += blockDim.x) {
    stacked_data[target] = stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t ChainBStackedToCaspar(const double* stacked_data, double* cas_data,
                                  const unsigned int cas_stride, const unsigned int cas_offset,
                                  const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ChainBStackedToCaspar_kernel<<<num_blocks, block_size>>>(stacked_data, cas_data, cas_stride,
                                                           cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t ChainBCasparToStacked(const double* cas_data, double* stacked_data,
                                  const unsigned int cas_stride, const unsigned int cas_offset,
                                  const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  ChainBCasparToStacked_kernel<<<num_blocks, block_size>>>(cas_data, stacked_data, cas_stride,
                                                           cas_offset, num_objects);

  return cudaGetLastError();
}

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

__global__ __launch_bounds__(block_size, 1) void Matrix22StackedToCaspar_kernel(
    const double* const __restrict__ stacked_data, double* const __restrict__ cas_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 4];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4; target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 4] = stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Matrix22CasparToStacked_kernel(
    const double* const __restrict__ cas_data, double* const __restrict__ stacked_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 4];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 4;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 4 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 4; target += blockDim.x) {
    stacked_data[target] = stacked_data_local[target - (blockIdx.x * blockDim.x) * 4];
  }
}

cudaError_t Matrix22StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix22StackedToCaspar_kernel<<<num_blocks, block_size>>>(stacked_data, cas_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Matrix22CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix22CasparToStacked_kernel<<<num_blocks, block_size>>>(cas_data, stacked_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void Matrix23StackedToCaspar_kernel(
    const double* const __restrict__ stacked_data, double* const __restrict__ cas_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 6];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 6 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 6; target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 6] = stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 6;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Matrix23CasparToStacked_kernel(
    const double* const __restrict__ cas_data, double* const __restrict__ stacked_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 6];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 6;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 6 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 6; target += blockDim.x) {
    stacked_data[target] = stacked_data_local[target - (blockIdx.x * blockDim.x) * 6];
  }
}

cudaError_t Matrix23StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix23StackedToCaspar_kernel<<<num_blocks, block_size>>>(stacked_data, cas_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Matrix23CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix23CasparToStacked_kernel<<<num_blocks, block_size>>>(cas_data, stacked_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void Matrix31StackedToCaspar_kernel(
    const double* const __restrict__ stacked_data, double* const __restrict__ cas_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3; target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 3] = stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Matrix31CasparToStacked_kernel(
    const double* const __restrict__ cas_data, double* const __restrict__ stacked_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 3];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 3;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[2] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 3 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 3; target += blockDim.x) {
    stacked_data[target] = stacked_data_local[target - (blockIdx.x * blockDim.x) * 3];
  }
}

cudaError_t Matrix31StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix31StackedToCaspar_kernel<<<num_blocks, block_size>>>(stacked_data, cas_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Matrix31CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix31CasparToStacked_kernel<<<num_blocks, block_size>>>(cas_data, stacked_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

__global__ __launch_bounds__(block_size, 1) void Matrix33StackedToCaspar_kernel(
    const double* const __restrict__ stacked_data, double* const __restrict__ cas_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 9];

  for (unsigned int target = (blockIdx.x * blockDim.x) * 9 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 9; target += blockDim.x) {
    stacked_data_local[target - (blockIdx.x * blockDim.x) * 9] = stacked_data[target];
  }

  __syncthreads();

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 9;
    double* out_ptr;
    data[0] = stacked_local_ptr[0];
    data[1] = stacked_local_ptr[1];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[2];
    data[1] = stacked_local_ptr[3];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[4];
    data[1] = stacked_local_ptr[5];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[6];
    data[1] = stacked_local_ptr[7];

    out_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    reinterpret_cast<double2*>(out_ptr)[0] = reinterpret_cast<double2*>(data)[0];
    data[0] = stacked_local_ptr[8];

    out_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 8 * cas_stride;
    out_ptr[0] = data[0];
  }
}

__global__ __launch_bounds__(block_size, 1) void Matrix33CasparToStacked_kernel(
    const double* const __restrict__ cas_data, double* const __restrict__ stacked_data,
    const unsigned int cas_stride, const unsigned int cas_offset, const unsigned int num_objects) {
  const unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ double stacked_data_local[block_size * 9];

  if (global_thread_idx < num_objects) {
    double data[4] = {0, 0, 0, 0};
    double* stacked_local_ptr = stacked_data_local + threadIdx.x * 9;
    const double* in_ptr;
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 0 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[0] = data[0];
    stacked_local_ptr[1] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 2 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[2] = data[0];
    stacked_local_ptr[3] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 4 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[4] = data[0];
    stacked_local_ptr[5] = data[1];
    in_ptr = cas_data + 2 * (global_thread_idx + cas_offset) + 6 * cas_stride;
    reinterpret_cast<double2*>(data)[0] = reinterpret_cast<const double2*>(in_ptr)[0];
    stacked_local_ptr[6] = data[0];
    stacked_local_ptr[7] = data[1];
    in_ptr = cas_data + 1 * (global_thread_idx + cas_offset) + 8 * cas_stride;
    data[0] = in_ptr[0];
    stacked_local_ptr[8] = data[0];
  }

  __syncthreads();

  for (unsigned int target = (blockIdx.x * blockDim.x) * 9 + threadIdx.x;
       target < min(num_objects, (blockIdx.x + 1) * blockDim.x) * 9; target += blockDim.x) {
    stacked_data[target] = stacked_data_local[target - (blockIdx.x * blockDim.x) * 9];
  }
}

cudaError_t Matrix33StackedToCaspar(const double* stacked_data, double* cas_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix33StackedToCaspar_kernel<<<num_blocks, block_size>>>(stacked_data, cas_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

cudaError_t Matrix33CasparToStacked(const double* cas_data, double* stacked_data,
                                    const unsigned int cas_stride, const unsigned int cas_offset,
                                    const unsigned int num_objects) {
  const int num_blocks = (num_objects + block_size - 1) / block_size;

  Matrix33CasparToStacked_kernel<<<num_blocks, block_size>>>(cas_data, stacked_data, cas_stride,
                                                             cas_offset, num_objects);

  return cudaGetLastError();
}

}  // namespace caspar